from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.run_cifar100_fixed_rank_head import FeatureData, SplitData, load_feature_data, split_features
from src.metrics import compute_metrics_stiefel
from src.manifolds.stiefel import (
    imuon_direction,
    rgd_direction,
    spel_direction,
    stiefel_polar,
    stiefel_qr,
)
from src.utils import DEFAULT_DTYPE, ensure_dir, format_lr, hardware_info, save_json, setup_seed, write_csv


torch.set_default_dtype(DEFAULT_DTYPE)


DEFAULT_METHODS = ["rgd", "imuon", "spel"]
DEFAULT_LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


@dataclass(frozen=True)
class StiefelState:
    x: torch.Tensor
    bias: torch.Tensor


METHODS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "rgd": rgd_direction,
    "imuon": imuon_direction,
    "spel": spel_direction,
}


def cosine_scores(state: StiefelState, features: torch.Tensor) -> torch.Tensor:
    return features @ state.x


def class_scores_from_state(state: StiefelState, features: torch.Tensor, prototypes_per_class: int) -> torch.Tensor:
    scores = cosine_scores(state, features)
    ppc = int(prototypes_per_class)
    if ppc == 1:
        return scores
    n_classes = int(state.bias.numel())
    if scores.shape[1] != n_classes * ppc:
        raise ValueError(f"expected {n_classes * ppc} prototype columns, got {scores.shape[1]}")
    return scores.reshape(scores.shape[0], n_classes, ppc).amax(dim=2)


def logits_from_state(
    state: StiefelState,
    features: torch.Tensor,
    logit_scale: float,
    use_bias: bool,
    prototypes_per_class: int = 1,
) -> torch.Tensor:
    logits = float(logit_scale) * class_scores_from_state(state, features, prototypes_per_class)
    if use_bias:
        logits = logits + state.bias
    return logits


def margin_logits(
    state: StiefelState,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_scale: float,
    use_bias: bool,
    margin_type: str,
    margin: float,
    prototypes_per_class: int = 1,
) -> torch.Tensor:
    scores = class_scores_from_state(state, features, prototypes_per_class)
    if margin_type != "none" and float(margin) > 0.0:
        target = scores.gather(1, labels[:, None]).squeeze(1)
        if margin_type == "cosface":
            target_m = target - float(margin)
        elif margin_type == "arcface":
            theta = torch.acos(target.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            target_m = torch.cos(theta + float(margin))
        else:
            raise ValueError(f"unknown margin_type={margin_type!r}")
        scores = scores.clone()
        scores.scatter_(1, labels[:, None], target_m[:, None])
    logits = float(logit_scale) * scores
    if use_bias:
        logits = logits + state.bias
    return logits


def objective_loss(
    state: StiefelState,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_scale: float,
    use_bias: bool,
    bias_weight_decay: float,
    margin_type: str,
    margin: float,
    prototypes_per_class: int = 1,
) -> torch.Tensor:
    logits = margin_logits(
        state,
        features,
        labels,
        logit_scale=logit_scale,
        use_bias=use_bias,
        margin_type=margin_type,
        margin=margin,
        prototypes_per_class=prototypes_per_class,
    )
    ce = F.cross_entropy(logits, labels)
    if use_bias and bias_weight_decay > 0.0:
        ce = ce + 0.5 * float(bias_weight_decay) * torch.sum(state.bias * state.bias)
    return ce


def evaluate_split(
    state: StiefelState,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_scale: float,
    use_bias: bool,
    batch_size: int,
    prototypes_per_class: int = 1,
) -> dict[str, float]:
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, int(labels.numel()), batch_size):
            end = min(start + batch_size, int(labels.numel()))
            feat = features[start:end]
            lab = labels[start:end]
            logits = logits_from_state(state, feat, logit_scale, use_bias, prototypes_per_class)
            loss_sum += float(F.cross_entropy(logits, lab, reduction="sum").item())
            correct += int((torch.argmax(logits, dim=1) == lab).sum().item())
            total += int(lab.numel())
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}


def _take_per_class(features: torch.Tensor, labels: torch.Tensor, per_class: int | None, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if per_class is None:
        return features, labels
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    labels_np = labels.detach().cpu().numpy()
    for c in sorted(np.unique(labels_np).tolist()):
        idx = np.flatnonzero(labels_np == c)
        take = min(int(per_class), int(idx.size))
        if take > 0:
            chosen.extend(int(i) for i in rng.choice(idx, size=take, replace=False))
    chosen = sorted(chosen)
    idx_t = torch.as_tensor(chosen, dtype=torch.long, device=features.device)
    return features.index_select(0, idx_t), labels.index_select(0, idx_t)


def subsample_split(
    split: SplitData,
    *,
    train_per_class: int | None,
    val_per_class: int | None,
    test_per_class: int | None,
    seed: int,
) -> SplitData:
    train_features, train_labels = _take_per_class(split.train_features, split.train_labels, train_per_class, seed + 11)
    val_features, val_labels = _take_per_class(split.val_features, split.val_labels, val_per_class, seed + 17)
    test_features, test_labels = _take_per_class(split.test_features, split.test_labels, test_per_class, seed + 23)
    return SplitData(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        split_indices=split.split_indices,
    )


def add_feature_noise_split(split: SplitData, *, std: float, seed: int, renormalize: bool = True) -> SplitData:
    if float(std) <= 0.0:
        return split

    def add_noise(features: torch.Tensor, offset: int) -> torch.Tensor:
        gen = torch.Generator(device=features.device).manual_seed(seed + offset)
        noisy = features + float(std) * torch.randn(features.shape, generator=gen, device=features.device, dtype=features.dtype)
        if renormalize:
            noisy = F.normalize(noisy, p=2, dim=1)
        return noisy

    return SplitData(
        train_features=add_noise(split.train_features, 101),
        train_labels=split.train_labels,
        val_features=add_noise(split.val_features, 103),
        val_labels=split.val_labels,
        test_features=add_noise(split.test_features, 107),
        test_labels=split.test_labels,
        split_indices=split.split_indices,
    )


def init_stiefel_classifier(
    split: SplitData,
    feature_dim: int,
    n_classes: int,
    seed: int,
    device: torch.device,
    *,
    prototypes_per_class: int = 1,
) -> StiefelState:
    ppc = int(prototypes_per_class)
    means = torch.zeros((feature_dim, n_classes), dtype=DEFAULT_DTYPE, device=device)
    counts = torch.zeros((n_classes,), dtype=DEFAULT_DTYPE, device=device)
    means.index_add_(1, split.train_labels, split.train_features.transpose(0, 1))
    counts.index_add_(0, split.train_labels, torch.ones_like(split.train_labels, dtype=DEFAULT_DTYPE))
    means = means / counts.clamp_min(1.0).unsqueeze(0)
    g = torch.Generator(device=device).manual_seed(seed + 90_001)
    if ppc == 1:
        proto = means + 1e-3 * torch.randn(means.shape, generator=g, device=device, dtype=DEFAULT_DTYPE)
    else:
        proto = torch.zeros((feature_dim, n_classes * ppc), dtype=DEFAULT_DTYPE, device=device)
        for c in range(n_classes):
            idx = torch.nonzero(split.train_labels == c, as_tuple=False).flatten()
            for j in range(ppc):
                col = c * ppc + j
                base = means[:, c]
                if idx.numel() > 0:
                    sample_pos = torch.randint(idx.numel(), (1,), generator=g, device=device).item()
                    sample = split.train_features[idx[int(sample_pos)]]
                    base = 0.7 * base + 0.3 * sample
                proto[:, col] = base
        proto = proto + 1e-3 * torch.randn(proto.shape, generator=g, device=device, dtype=DEFAULT_DTYPE)
    x = stiefel_polar(proto)
    bias = torch.zeros((n_classes,), dtype=DEFAULT_DTYPE, device=device)
    return StiefelState(x=x.detach(), bias=bias)


def run_one_training(
    split: SplitData,
    init_state: StiefelState,
    *,
    method: str,
    lr: float,
    max_epochs: int,
    batch_size: int,
    eval_batch_size: int,
    logit_scale: float,
    margin_type: str,
    margin: float,
    retraction: str,
    use_bias: bool,
    bias_lr: float,
    bias_weight_decay: float,
    prototypes_per_class: int,
    batch_seed_base: int,
) -> tuple[StiefelState, list[dict[str, Any]], dict[str, Any]]:
    direction_fn = METHODS[method]
    state = StiefelState(x=init_state.x.clone(), bias=init_state.bias.clone())
    logs: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    n_train = int(split.train_labels.numel())
    retract_fn = stiefel_polar if retraction == "polar" else stiefel_qr

    for epoch in range(1, max_epochs + 1):
        gen = torch.Generator(device=split.train_features.device).manual_seed(batch_seed_base + epoch)
        perm = torch.randperm(n_train, generator=gen, device=split.train_features.device)
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        batch_metrics: list[dict[str, float]] = []

        for start in range(0, n_train, batch_size):
            idx = perm[start : min(start + batch_size, n_train)]
            feat = split.train_features.index_select(0, idx)
            lab = split.train_labels.index_select(0, idx)
            x_var = state.x.detach().clone().requires_grad_(True)
            b_var = state.bias.detach().clone().requires_grad_(use_bias)
            var_state = StiefelState(x=x_var, bias=b_var)
            loss = objective_loss(
                var_state,
                feat,
                lab,
                logit_scale=logit_scale,
                use_bias=use_bias,
                bias_weight_decay=bias_weight_decay,
                margin_type=margin_type,
                margin=margin,
                prototypes_per_class=prototypes_per_class,
            )
            loss.backward()
            egrad = x_var.grad.detach()
            direction = direction_fn(state.x, egrad)
            batch_metrics.append(compute_metrics_stiefel(state.x, egrad, direction, method=method))

            with torch.no_grad():
                x_new = retract_fn(state.x - float(lr) * direction)
                if use_bias:
                    b_grad = b_var.grad.detach()
                    b_new = state.bias - float(bias_lr) * b_grad
                else:
                    b_new = state.bias
                state = StiefelState(x=x_new.detach(), bias=b_new.detach())
                logits = logits_from_state(state, feat, logit_scale, use_bias, prototypes_per_class)
                train_loss_sum += float(loss.item()) * int(lab.numel())
                train_correct += int((torch.argmax(logits, dim=1) == lab).sum().item())
                train_count += int(lab.numel())

        train_loss = train_loss_sum / max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)
        val_eval = evaluate_split(
            state,
            split.val_features,
            split.val_labels,
            logit_scale=logit_scale,
            use_bias=use_bias,
            batch_size=eval_batch_size,
            prototypes_per_class=prototypes_per_class,
        )
        test_eval = evaluate_split(
            state,
            split.test_features,
            split.test_labels,
            logit_scale=logit_scale,
            use_bias=use_bias,
            batch_size=eval_batch_size,
            prototypes_per_class=prototypes_per_class,
        )
        avg_metrics = {key: float(np.mean([row[key] for row in batch_metrics])) for key in batch_metrics[0]}
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_eval["loss"],
            "val_acc": val_eval["acc"],
            "test_loss": test_eval["loss"],
            "test_acc": test_eval["acc"],
            "wall_time_sec": time.perf_counter() - t0,
        }
        row.update(avg_metrics)
        logs.append(row)

    final = logs[-1]
    summary = {
        "method": method,
        "lr": lr,
        "final_train_loss": final["train_loss"],
        "final_train_acc": final["train_acc"],
        "final_val_loss": final["val_loss"],
        "final_val_acc": final["val_acc"],
        "final_test_loss": final["test_loss"],
        "final_test_acc": final["test_acc"],
        "final_wall_time_sec": final["wall_time_sec"],
        "final_rgrad_fro": final["rgrad_fro"],
        "final_skew_grad_nuc": final["skew_grad_nuc"],
        "final_normal_grad_nuc": final["normal_grad_nuc"],
        "final_skew_normal_nuc_ratio": final["skew_normal_nuc_ratio"],
        "final_direction_tangent_violation": final["direction_tangent_violation"],
        "final_spel_direction_tangent_violation": final["spel_direction_tangent_violation"],
        "final_direction_skew_fro": final["direction_skew_fro"],
        "final_direction_normal_fro": final["direction_normal_fro"],
    }
    return state, logs, summary


def save_run_outputs(run_dir: Path, config: dict[str, Any], logs: list[dict[str, Any]], final_state: StiefelState, summary: dict[str, Any]) -> None:
    ensure_dir(run_dir)
    save_json(run_dir / "config.json", config)
    write_csv(run_dir / "metrics.csv", logs)
    save_json(run_dir / "summary.json", summary)
    torch.save({"x": final_state.x.detach().cpu(), "bias": final_state.bias.detach().cpu()}, run_dir / "final_model.pt")


def summarize_sweep(results_dir: Path, methods: list[str], seeds: list[int], lr_grid: list[float]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    best: dict[str, float] = {}
    for method in methods:
        candidates: list[dict[str, Any]] = []
        for lr in lr_grid:
            vals: dict[str, list[float]] = {
                "final_val_acc": [],
                "final_val_loss": [],
                "final_test_acc": [],
                "final_test_loss": [],
                "final_spel_direction_tangent_violation": [],
                "final_direction_tangent_violation": [],
                "final_skew_normal_nuc_ratio": [],
            }
            for seed in seeds:
                path = results_dir / f"seed{seed}" / method / f"lr_{format_lr(lr)}" / "summary.json"
                if not path.exists():
                    continue
                import json

                payload = json.loads(path.read_text())
                for key in vals:
                    vals[key].append(float(payload[key]))
            if vals["final_val_acc"]:
                row = {
                    "method": method,
                    "lr": float(lr),
                    "n_seeds": len(vals["final_val_acc"]),
                    "mean_final_val_acc": float(np.mean(vals["final_val_acc"])),
                    "std_final_val_acc": float(np.std(vals["final_val_acc"], ddof=1)) if len(vals["final_val_acc"]) > 1 else 0.0,
                    "mean_final_val_loss": float(np.mean(vals["final_val_loss"])),
                    "mean_final_test_acc": float(np.mean(vals["final_test_acc"])),
                    "std_final_test_acc": float(np.std(vals["final_test_acc"], ddof=1)) if len(vals["final_test_acc"]) > 1 else 0.0,
                    "mean_final_test_loss": float(np.mean(vals["final_test_loss"])),
                    "mean_final_spel_direction_tangent_violation": float(np.mean(vals["final_spel_direction_tangent_violation"])),
                    "mean_final_direction_tangent_violation": float(np.mean(vals["final_direction_tangent_violation"])),
                    "mean_final_skew_normal_nuc_ratio": float(np.mean(vals["final_skew_normal_nuc_ratio"])),
                }
                rows.append(row)
                candidates.append(row)
        if candidates:
            chosen = max(candidates, key=lambda r: (r["mean_final_val_acc"], -r["mean_final_val_loss"], r["n_seeds"], -r["lr"]))
            best[method] = float(chosen["lr"])
    return rows, best


def aggregate_best(results_dir: Path, best_lrs: dict[str, float], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, lr in best_lrs.items():
        vals: dict[str, list[float]] = {
            "final_val_acc": [],
            "final_val_loss": [],
            "final_test_acc": [],
            "final_test_loss": [],
            "final_spel_direction_tangent_violation": [],
            "final_direction_tangent_violation": [],
            "final_skew_normal_nuc_ratio": [],
        }
        for seed in seeds:
            path = results_dir / f"seed{seed}" / method / f"lr_{format_lr(lr)}" / "summary.json"
            if not path.exists():
                continue
            import json

            payload = json.loads(path.read_text())
            for key in vals:
                vals[key].append(float(payload[key]))
        rows.append(
            {
                "method": method,
                "best_lr": lr,
                "n_seeds": len(vals["final_test_acc"]),
                "mean_final_val_acc": float(np.mean(vals["final_val_acc"])) if vals["final_val_acc"] else None,
                "std_final_val_acc": float(np.std(vals["final_val_acc"], ddof=1)) if len(vals["final_val_acc"]) > 1 else 0.0 if vals["final_val_acc"] else None,
                "mean_final_test_acc": float(np.mean(vals["final_test_acc"])) if vals["final_test_acc"] else None,
                "std_final_test_acc": float(np.std(vals["final_test_acc"], ddof=1)) if len(vals["final_test_acc"]) > 1 else 0.0 if vals["final_test_acc"] else None,
                "mean_final_test_loss": float(np.mean(vals["final_test_loss"])) if vals["final_test_loss"] else None,
                "mean_final_spel_direction_tangent_violation": float(np.mean(vals["final_spel_direction_tangent_violation"])) if vals["final_spel_direction_tangent_violation"] else None,
                "mean_final_direction_tangent_violation": float(np.mean(vals["final_direction_tangent_violation"])) if vals["final_direction_tangent_violation"] else None,
                "mean_final_skew_normal_nuc_ratio": float(np.mean(vals["final_skew_normal_nuc_ratio"])) if vals["final_skew_normal_nuc_ratio"] else None,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stiefel classifier on frozen CIFAR-100 features, with SPEL and iMuon baselines.")
    parser.add_argument("--data-root", default="data/cifar100")
    parser.add_argument("--feature-cache-dir", default="data/cifar100/features")
    parser.add_argument("--results-dir", default="results/cifar100_stiefel_classifier")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=sorted(METHODS.keys()))
    parser.add_argument("--lr-grid", type=float, nargs="+", default=DEFAULT_LR_GRID)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--val-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--logit-scale", type=float, default=16.0)
    parser.add_argument("--prototypes-per-class", type=int, default=1)
    parser.add_argument("--margin-type", choices=["none", "cosface", "arcface"], default="none")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--retraction", choices=["polar", "qr"], default="polar")
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-weight-decay", type=float, default=0.0)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--feature-l2-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--feature-noise-renormalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use a small real-data subset and short LR grid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 5)
        args.lr_grid = [3e-3, 1e-2, 3e-2]
        args.train_per_class = args.train_per_class or 20
        args.val_per_class = args.val_per_class or 5
        args.test_per_class = args.test_per_class or 10

    setup_seed(args.seed)
    device = torch.device(args.device)
    data: FeatureData = load_feature_data(args, device)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    results_dir = ensure_dir(args.results_dir)

    save_json(
        results_dir / "run_config.json",
        {
            "seeds": seeds,
            "methods": args.methods,
            "lr_grid": args.lr_grid,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "validation_fraction": args.validation_fraction,
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "test_per_class": args.test_per_class,
            "logit_scale": args.logit_scale,
            "prototypes_per_class": args.prototypes_per_class,
            "margin_type": args.margin_type,
            "margin": args.margin,
            "retraction": args.retraction,
            "use_bias": args.use_bias,
            "bias_lr": args.bias_lr,
            "bias_weight_decay": args.bias_weight_decay,
            "data_source": data.source,
            "feature_dim": data.feature_dim,
            "n_classes": data.n_classes,
            "backbone": args.backbone,
            "feature_l2_normalize": args.feature_l2_normalize,
            "feature_noise_std": args.feature_noise_std,
            "feature_noise_renormalize": args.feature_noise_renormalize,
            "hardware": hardware_info(),
        },
    )

    n_proto = int(data.n_classes * args.prototypes_per_class)
    if data.feature_dim < n_proto:
        raise ValueError(f"Stiefel classifier needs feature_dim >= n_classes * prototypes_per_class, got {data.feature_dim} < {n_proto}")

    for seed in seeds:
        split = split_features(data, seed=seed, validation_fraction=args.validation_fraction, device=device)
        split = subsample_split(
            split,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
            seed=seed,
        )
        split = add_feature_noise_split(
            split,
            std=args.feature_noise_std,
            seed=900_000 + 10_000 * seed,
            renormalize=args.feature_noise_renormalize,
        )
        init_state = init_stiefel_classifier(
            split,
            data.feature_dim,
            data.n_classes,
            seed,
            device,
            prototypes_per_class=args.prototypes_per_class,
        )
        torch.save({"x": init_state.x.detach().cpu(), "bias": init_state.bias.detach().cpu()}, results_dir / f"init_seed{seed}.pt")
        for method in args.methods:
            for lr in args.lr_grid:
                run_dir = ensure_dir(results_dir / f"seed{seed}" / method / f"lr_{format_lr(lr)}")
                config = {
                    "seed": seed,
                    "method": method,
                    "lr": lr,
                    "max_epochs": args.max_epochs,
                    "n_train_examples": int(split.train_labels.numel()),
                    "n_val_examples": int(split.val_labels.numel()),
                    "n_test_examples": int(split.test_labels.numel()),
                    "feature_dim": data.feature_dim,
                    "n_classes": data.n_classes,
                    "n_prototypes": n_proto,
                    "prototypes_per_class": args.prototypes_per_class,
                    "logit_scale": args.logit_scale,
                    "margin_type": args.margin_type,
                    "margin": args.margin,
                    "retraction": args.retraction,
                    "use_bias": args.use_bias,
                    "feature_noise_std": args.feature_noise_std,
                    "feature_noise_renormalize": args.feature_noise_renormalize,
                }
                final_state, logs, summary = run_one_training(
                    split,
                    init_state,
                    method=method,
                    lr=float(lr),
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    logit_scale=args.logit_scale,
                    margin_type=args.margin_type,
                    margin=args.margin,
                    retraction=args.retraction,
                    use_bias=args.use_bias,
                    bias_lr=args.bias_lr,
                    bias_weight_decay=args.bias_weight_decay,
                    prototypes_per_class=args.prototypes_per_class,
                    batch_seed_base=700_000 + 10_000 * seed,
                )
                save_run_outputs(run_dir, config, logs, final_state, summary)

    sweep_rows, best = summarize_sweep(results_dir, args.methods, seeds, args.lr_grid)
    write_csv(results_dir / "lr_sweep_summary.csv", sweep_rows)
    save_json(results_dir / "global_best_lr.json", best)
    write_csv(results_dir / "mean_test_accuracy_table.csv", aggregate_best(results_dir, best, seeds))


if __name__ == "__main__":
    main()
