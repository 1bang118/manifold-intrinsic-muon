from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_cifar100_fixed_rank_head import (
    DEFAULT_DTYPE,
    METHODS,
    format_lr,
    hardware_info,
    init_head,
    load_feature_data,
    run_one_training,
    save_json,
    save_run_outputs,
    save_shared_artifacts,
    split_features,
    write_csv,
)
from src.utils import ensure_dir, setup_seed


def parse_method_lrs(items: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected METHOD=LR, got {item!r}")
        method, lr_text = item.split("=", 1)
        method = method.strip()
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choices: {sorted(METHODS)}")
        values = [float(v) for v in lr_text.replace(",", " ").split()]
        if not values:
            raise ValueError(f"No learning rates provided for {method!r}")
        out.setdefault(method, []).extend(values)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CIFAR-100 fixed-rank head with one selected LR per method."
    )
    parser.add_argument("--data-root", default="data/cifar100")
    parser.add_argument("--feature-cache-dir", default="data/cifar100/features")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--gauge-alpha", type=float, default=1.0)
    parser.add_argument("--init-mode", choices=["random", "class_mean_svd"], default="random")
    parser.add_argument(
        "--method-lr",
        nargs="+",
        required=True,
        help="Selected/candidate learning rates as METHOD=LR or METHOD=LR1,LR2.",
    )
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bias-weight-decay", type=float, default=1e-4)
    parser.add_argument("--bias-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--feature-l2-normalize", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def aggregate_candidate_rows(results_dir: Path, rank: int, method_lrs: dict[str, list[float]], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = [
        "final_val_acc",
        "final_val_loss",
        "final_test_acc",
        "final_test_loss",
        "final_train_acc",
        "final_train_loss",
        "final_rgrad_norm",
        "final_dual_norm_H",
        "final_Z_norm_sq",
        "final_kappa_left",
        "final_kappa_right",
        "final_factor_norm_sq",
        "final_delta_geom",
        "final_raw_vs_whitened_cosine",
        "final_wall_time_sec",
    ]
    for method, lrs in method_lrs.items():
        for lr in lrs:
            values: dict[str, list[float]] = {key: [] for key in keys}
            for seed in seeds:
                path = results_dir / f"seed{seed}" / f"rank{rank}" / method / f"lr_{format_lr(lr)}" / "summary.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text())
                for key in keys:
                    values[key].append(float(payload[key]))
            row: dict[str, Any] = {
                "rank": rank,
                "method": method,
                "candidate_lr": lr,
                "n_seeds": len(values["final_test_acc"]),
            }
            for key, arr in values.items():
                row[f"mean_{key}"] = float(np.mean(arr)) if arr else None
                row[f"std_{key}"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0 if arr else None
            rows.append(row)
    return rows


def select_best_rows(candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        if row["n_seeds"] <= 0:
            continue
        by_method.setdefault(str(row["method"]), []).append(row)

    rows: list[dict[str, Any]] = []
    selected_lrs: dict[str, float] = {}
    for method, candidates in by_method.items():
        best = max(
            candidates,
            key=lambda row: (
                row["mean_final_val_acc"],
                -row["mean_final_val_loss"],
                row["n_seeds"],
                -row["candidate_lr"],
            ),
        )
        selected_lrs[method] = float(best["candidate_lr"])
        out = dict(best)
        out["selected_lr"] = out.pop("candidate_lr")
        rows.append(out)
    return rows, selected_lrs


def aggregate_method_rows(results_dir: Path, rank: int, method_lrs: dict[str, list[float]], seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    candidate_rows = aggregate_candidate_rows(results_dir, rank, method_lrs, seeds)
    best_rows, selected_lrs = select_best_rows(candidate_rows)
    return candidate_rows, best_rows, selected_lrs


def aggregate_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method = {row["method"]: row for row in rows}
    pairs = [
        ("riemannian_gd", "euclidean_gd", "frobenius"),
        ("scaled_muon", "euclidean_muon", "spectral"),
        ("scaled_numuon", "euclidean_numuon", "nuclear"),
    ]
    out: list[dict[str, Any]] = []
    for left, right, label in pairs:
        if left not in by_method or right not in by_method:
            continue
        a = by_method[left]
        b = by_method[right]
        out.append(
            {
                "pair": label,
                "intrinsic_method": left,
                "euclidean_method": right,
                "intrinsic_lr": a["selected_lr"],
                "euclidean_lr": b["selected_lr"],
                "n_intrinsic_seeds": a["n_seeds"],
                "n_euclidean_seeds": b["n_seeds"],
                "mean_test_acc_gap": a["mean_final_test_acc"] - b["mean_final_test_acc"]
                if a["mean_final_test_acc"] is not None and b["mean_final_test_acc"] is not None
                else None,
                "mean_val_acc_gap": a["mean_final_val_acc"] - b["mean_final_val_acc"]
                if a["mean_final_val_acc"] is not None and b["mean_final_val_acc"] is not None
                else None,
                "mean_test_loss_gap": a["mean_final_test_loss"] - b["mean_final_test_loss"]
                if a["mean_final_test_loss"] is not None and b["mean_final_test_loss"] is not None
                else None,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    torch.set_default_dtype(DEFAULT_DTYPE)
    device = torch.device(args.device)
    results_dir = ensure_dir(args.results_dir)
    selected_lrs = parse_method_lrs(args.method_lr)
    methods = list(selected_lrs)
    data = load_feature_data(args, device)

    save_json(
        results_dir / "run_config.json",
        {
            "data_source": data.source,
            "feature_dim": data.feature_dim,
            "n_classes": data.n_classes,
            "n_train_examples": int(data.train_labels.numel()),
            "n_test_examples": int(data.test_labels.numel()),
            "methods": methods,
            "candidate_lrs": selected_lrs,
            "rank": args.rank,
            "gauge_alpha": args.gauge_alpha,
            "init_mode": args.init_mode,
            "seeds": args.seeds,
            "max_epochs": args.max_epochs,
            "validation_fraction": args.validation_fraction,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "feature_batch_size": args.feature_batch_size,
            "weight_decay": args.weight_decay,
            "bias_weight_decay": args.bias_weight_decay,
            "bias_lr_multiplier": args.bias_lr_multiplier,
            "backbone": args.backbone,
            "feature_l2_normalize": args.feature_l2_normalize,
            "synthetic": args.synthetic,
            "device": str(device),
            "hardware": hardware_info(),
        },
    )
    save_json(results_dir / "candidate_lrs.json", selected_lrs)

    shared_dir = ensure_dir(results_dir / "shared")
    for seed in args.seeds:
        split = split_features(data, seed=seed, validation_fraction=args.validation_fraction, device=device)
        init_state = init_head(
            data.feature_dim,
            data.n_classes,
            args.rank,
            seed,
            device,
            gauge_alpha=args.gauge_alpha,
            init_mode=args.init_mode,
            split=split,
        )
        save_shared_artifacts(shared_dir, args.rank, seed, split, init_state)
        for method in methods:
            for lr in selected_lrs[method]:
                lr = float(lr)
                run_dir = ensure_dir(results_dir / f"seed{seed}" / f"rank{args.rank}" / method / f"lr_{format_lr(lr)}")
                summary_path = run_dir / "summary.json"
                if args.resume and summary_path.exists():
                    print(f"[skip] seed={seed} method={method} lr={lr}", flush=True)
                    continue
                config = {
                    "seed": seed,
                    "rank": args.rank,
                    "gauge_alpha": args.gauge_alpha,
                    "init_mode": args.init_mode,
                    "method": method,
                    "lr": lr,
                    "max_epochs": args.max_epochs,
                    "validation_fraction": args.validation_fraction,
                    "data_source": data.source,
                    "feature_dim": data.feature_dim,
                    "n_classes": data.n_classes,
                    "n_train_examples": int(split.train_labels.numel()),
                    "n_val_examples": int(split.val_labels.numel()),
                    "n_test_examples": int(split.test_labels.numel()),
                    "norm": METHODS[method]["norm"],
                    "geometry": METHODS[method]["geometry"],
                    "backbone": args.backbone,
                    "weight_decay": args.weight_decay,
                    "bias_weight_decay": args.bias_weight_decay,
                    "bias_lr_multiplier": args.bias_lr_multiplier,
                }
                print(f"[run] seed={seed} method={method} lr={lr}", flush=True)
                final_state, logs, summary = run_one_training(
                    split,
                    init_state,
                    method=method,
                    lr=lr,
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    weight_decay=args.weight_decay,
                    bias_weight_decay=args.bias_weight_decay,
                    bias_lr_multiplier=args.bias_lr_multiplier,
                    eval_batch_size=args.eval_batch_size,
                    batch_seed_base=500_000 + 10_000 * seed + 100 * args.rank,
                )
                save_run_outputs(run_dir, config, shared_dir / f"init_seed{seed}_rank{args.rank}.pt", logs, final_state, summary)
                print(
                    f"[done] seed={seed} method={method} lr={lr} "
                    f"val={summary['final_val_acc']:.4f} test={summary['final_test_acc']:.4f}",
                    flush=True,
                )

    candidate_rows, rows, best_lrs = aggregate_method_rows(results_dir, args.rank, selected_lrs, args.seeds)
    write_csv(results_dir / "candidate_lr_summary.csv", candidate_rows)
    write_csv(results_dir / "mean_test_accuracy_table.csv", rows)
    write_csv(results_dir / "pairwise_gap_summary.csv", aggregate_pairwise(rows))
    save_json(results_dir / "selected_lrs.json", best_lrs)
    print(f"[summary] wrote {results_dir / 'candidate_lr_summary.csv'}", flush=True)
    print(f"[summary] wrote {results_dir / 'mean_test_accuracy_table.csv'}", flush=True)
    print(f"[summary] wrote {results_dir / 'pairwise_gap_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
