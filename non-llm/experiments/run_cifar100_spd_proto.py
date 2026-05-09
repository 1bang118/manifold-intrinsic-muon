from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models

from src.manifolds.spd import (
    euclidean_gd_step,
    euclidean_muon_step,
    euclidean_numuon_step,
    retraction_exp,
    spd_muon_step,
    spd_numuon_step,
    spd_riemannian_gd_step,
)
from src.metrics import compute_metrics_spd
from src.objectives.spd_barycenter import arithmetic_mean, log_euclidean_mean, spd_distance_sq
from src.objectives.spd_joint_proto import batch_proto_scores, build_initial_prototypes, joint_proto_ce_loss
from src.utils import DEFAULT_DTYPE, ensure_dir, format_lr, hardware_info, project_spd, save_json, setup_seed, sym, write_csv


torch.set_default_dtype(DEFAULT_DTYPE)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


StepFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

DEFAULT_METHODS = [
    "riemannian_gd",
    "spd_muon",
    "spd_numuon",
    "euclidean_gd",
    "euclidean_muon",
    "euclidean_numuon",
]

ALL_ITERATIVE_METHODS: dict[str, dict[str, Any]] = {
    "riemannian_gd": {"step_fn": spd_riemannian_gd_step, "norm": "frobenius"},
    "spd_muon": {"step_fn": spd_muon_step, "norm": "spectral"},
    "spd_numuon": {"step_fn": spd_numuon_step, "norm": "nuclear"},
    "euclidean_gd": {"step_fn": euclidean_gd_step, "norm": "frobenius"},
    "euclidean_muon": {"step_fn": euclidean_muon_step, "norm": "spectral"},
    "euclidean_numuon": {"step_fn": euclidean_numuon_step, "norm": "nuclear"},
}

DEFAULT_LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SMOKE_LR_GRID = [1e-3, 1e-2]


@dataclass(frozen=True)
class CovarianceData:
    train_covs: torch.Tensor
    train_labels: torch.Tensor
    test_covs: torch.Tensor
    test_labels: torch.Tensor
    class_names: list[str]
    feature_layer: str
    projected_dim: int
    source: str


@dataclass(frozen=True)
class SplitData:
    train_covs: torch.Tensor
    train_labels: torch.Tensor
    val_covs: torch.Tensor
    val_labels: torch.Tensor
    test_covs: torch.Tensor
    test_labels: torch.Tensor
    train_indices: np.ndarray
    val_indices: np.ndarray


def _backbone_spec(name: str) -> tuple[Any, Any]:
    if name == "resnet18":
        return models.resnet18, models.ResNet18_Weights.DEFAULT
    if name == "resnet50":
        return models.resnet50, models.ResNet50_Weights.DEFAULT
    raise ValueError(f"unsupported backbone: {name}")


def _load_cifar100_raw_labels(root: Path, train: bool, label_granularity: str) -> tuple[list[int], list[str]]:
    base = root / "cifar-100-python"
    split_path = base / ("train" if train else "test")
    meta_path = base / "meta"
    with split_path.open("rb") as f:
        payload = pickle.load(f, encoding="bytes")
    with meta_path.open("rb") as f:
        meta = pickle.load(f, encoding="bytes")
    if label_granularity == "coarse":
        labels = [int(v) for v in payload[b"coarse_labels"]]
        names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in meta[b"coarse_label_names"]]
        return labels, names
    if label_granularity == "fine":
        labels = [int(v) for v in payload[b"fine_labels"]]
        names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in meta[b"fine_label_names"]]
        return labels, names
    raise ValueError(f"unsupported label_granularity={label_granularity!r}")


def _subset_indices_by_label(labels: list[int], per_class: int | None, seed: int) -> list[int]:
    if per_class is None:
        return list(range(len(labels)))
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=np.int64)
    selected: list[int] = []
    for class_idx in sorted(np.unique(labels_arr).tolist()):
        indices = np.flatnonzero(labels_arr == class_idx)
        take = min(int(per_class), int(indices.size))
        chosen = rng.choice(indices, size=take, replace=False)
        selected.extend(int(idx) for idx in np.sort(chosen))
    return sorted(selected)


def _build_feature_extractor(backbone: str, feature_layer: str, device: torch.device) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor], int]:
    model_fn, weights = _backbone_spec(backbone)
    backbone_model = model_fn(weights=weights)
    children = list(backbone_model.named_children())
    modules: list[torch.nn.Module] = []
    out_channels = 0
    for name, module in children:
        if name == "fc":
            break
        modules.append(module)
        if name == feature_layer:
            out_channels = int(module[-1].conv2.out_channels) if name.startswith("layer") else 64
            break
    if out_channels == 0:
        raise ValueError(f"feature_layer={feature_layer!r} not found in backbone {backbone}")
    extractor = torch.nn.Sequential(*modules).to(device)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad_(False)
    return weights.transforms(), extractor, out_channels


def _random_projector(in_dim: int, out_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    if out_dim > in_dim:
        raise ValueError(f"projected_dim={out_dim} exceeds input dim {in_dim}")
    g = torch.Generator(device=device).manual_seed(seed)
    mat = torch.randn((in_dim, out_dim), generator=g, device=device, dtype=DEFAULT_DTYPE)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q


def _feature_map_to_covariance(
    fmap: torch.Tensor,
    projector: torch.Tensor,
    shrinkage: float,
    eps: float,
) -> torch.Tensor:
    # fmap: (B, C, H, W) -> tokens: (B, HW, C)
    tokens = fmap.flatten(2).transpose(1, 2).to(dtype=DEFAULT_DTYPE)
    proj_tokens = tokens @ projector
    centered = proj_tokens - proj_tokens.mean(dim=1, keepdim=True)
    denom = max(int(centered.shape[1]) - 1, 1)
    cov = centered.transpose(1, 2) @ centered / float(denom)
    cov = sym(cov)
    if float(shrinkage) > 0.0:
        dim = cov.shape[-1]
        eye = torch.eye(dim, dtype=cov.dtype, device=cov.device).unsqueeze(0).expand(cov.shape[0], -1, -1)
        trace_term = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(dim)
        cov = (1.0 - float(shrinkage)) * cov + float(shrinkage) * trace_term[:, None, None] * eye
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device).unsqueeze(0).expand(cov.shape[0], -1, -1)
    return cov + float(eps) * eye


def _extract_cov_split(
    ds: torch.utils.data.Dataset[Any],
    indices: list[int],
    labels: list[int],
    *,
    transform_batch_size: int,
    num_workers: int,
    device: torch.device,
    transforms: Callable[[Any], torch.Tensor],
    extractor: torch.nn.Module,
    projector: torch.Tensor,
    shrinkage: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    subset = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(
        subset,
        batch_size=transform_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    covs: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device=device, dtype=DEFAULT_DTYPE)
            fmap = extractor(images)
            cov = _feature_map_to_covariance(fmap, projector, shrinkage=shrinkage, eps=eps)
            covs.append(cov.cpu().to(dtype=torch.float32))
        for start in range(0, len(indices), transform_batch_size):
            chunk = labels[start : start + transform_batch_size]
            label_rows.append(torch.tensor(chunk, dtype=torch.long))
    return torch.cat(covs, dim=0), torch.cat(label_rows, dim=0)


def load_or_extract_covariances(
    *,
    data_root: Path,
    cache_dir: Path,
    backbone: str,
    feature_layer: str,
    label_granularity: str,
    projected_dim: int,
    projection_seed: int,
    extraction_batch_size: int,
    num_workers: int,
    shrinkage: float,
    eps: float,
    device: torch.device,
    smoke: bool,
    smoke_train_per_class: int,
    smoke_test_per_class: int,
    train_per_class: int | None,
    test_per_class: int | None,
) -> CovarianceData:
    cache_name = f"cifar100_{backbone}_{feature_layer}_cov{projected_dim}_{label_granularity}"
    effective_train_per_class = train_per_class
    effective_test_per_class = test_per_class
    if smoke:
        if effective_train_per_class is None:
            effective_train_per_class = smoke_train_per_class
        if effective_test_per_class is None:
            effective_test_per_class = smoke_test_per_class
    if effective_train_per_class is not None or effective_test_per_class is not None:
        cache_name += f"_tr{effective_train_per_class if effective_train_per_class is not None else 'all'}"
        cache_name += f"_te{effective_test_per_class if effective_test_per_class is not None else 'all'}"
    cache_path = cache_dir / f"{cache_name}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return CovarianceData(
            train_covs=payload["train_covs"].to(dtype=torch.float32),
            train_labels=payload["train_labels"].long(),
            test_covs=payload["test_covs"].to(dtype=torch.float32),
            test_labels=payload["test_labels"].long(),
            class_names=[str(x) for x in payload["class_names"]],
            feature_layer=str(payload["feature_layer"]),
            projected_dim=int(payload["projected_dim"]),
            source=str(cache_path),
        )

    transforms, extractor, in_dim = _build_feature_extractor(backbone, feature_layer, device)
    projector = _random_projector(in_dim, projected_dim, projection_seed, device)
    train_ds = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transforms)
    test_ds = datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=transforms)
    train_labels, class_names = _load_cifar100_raw_labels(data_root, True, label_granularity)
    test_labels, _ = _load_cifar100_raw_labels(data_root, False, label_granularity)
    train_indices = _subset_indices_by_label(train_labels, effective_train_per_class, projection_seed + 11)
    test_indices = _subset_indices_by_label(test_labels, effective_test_per_class, projection_seed + 23)

    train_covs, train_label_tensor = _extract_cov_split(
        train_ds,
        train_indices,
        [train_labels[idx] for idx in train_indices],
        transform_batch_size=extraction_batch_size,
        num_workers=num_workers,
        device=device,
        transforms=transforms,
        extractor=extractor,
        projector=projector,
        shrinkage=shrinkage,
        eps=eps,
    )
    test_covs, test_label_tensor = _extract_cov_split(
        test_ds,
        test_indices,
        [test_labels[idx] for idx in test_indices],
        transform_batch_size=extraction_batch_size,
        num_workers=num_workers,
        device=device,
        transforms=transforms,
        extractor=extractor,
        projector=projector,
        shrinkage=shrinkage,
        eps=eps,
    )
    ensure_dir(cache_dir)
    torch.save(
        {
            "train_covs": train_covs,
            "train_labels": train_label_tensor,
            "test_covs": test_covs,
            "test_labels": test_label_tensor,
            "class_names": class_names,
            "feature_layer": feature_layer,
            "projected_dim": projected_dim,
            "projection_seed": projection_seed,
            "shrinkage": shrinkage,
            "eps": eps,
        },
        cache_path,
    )
    return CovarianceData(
        train_covs=train_covs,
        train_labels=train_label_tensor,
        test_covs=test_covs,
        test_labels=test_label_tensor,
        class_names=class_names,
        feature_layer=feature_layer,
        projected_dim=projected_dim,
        source=str(cache_path),
    )


def split_train_val(
    data: CovarianceData,
    *,
    seed: int,
    validation_fraction: float,
) -> SplitData:
    rng = np.random.default_rng(seed)
    labels_np = data.train_labels.numpy()
    class_ids = sorted(np.unique(labels_np).tolist())
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_idx in class_ids:
        indices = np.flatnonzero(labels_np == class_idx)
        perm = rng.permutation(indices)
        n_val = max(1, int(round(validation_fraction * len(indices))))
        n_val = min(n_val, len(indices) - 1)
        val_indices.extend(int(i) for i in np.sort(perm[:n_val]))
        train_indices.extend(int(i) for i in np.sort(perm[n_val:]))
    train_idx = np.asarray(sorted(train_indices), dtype=np.int64)
    val_idx = np.asarray(sorted(val_indices), dtype=np.int64)
    return SplitData(
        train_covs=data.train_covs[train_idx],
        train_labels=data.train_labels[train_idx],
        val_covs=data.train_covs[val_idx],
        val_labels=data.train_labels[val_idx],
        test_covs=data.test_covs,
        test_labels=data.test_labels,
        train_indices=train_idx,
        val_indices=val_idx,
    )


def _metric_from_predictions(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, float]:
    true_np = y_true.detach().cpu().numpy()
    pred_np = y_pred.detach().cpu().numpy()
    accuracy = float(np.mean(true_np == pred_np))

    labels = np.union1d(true_np, pred_np)
    label_to_idx = {int(label): idx for idx, label in enumerate(labels.tolist())}
    conf = np.zeros((labels.size, labels.size), dtype=np.float64)
    for true_label, pred_label in zip(true_np.tolist(), pred_np.tolist(), strict=False):
        conf[label_to_idx[int(true_label)], label_to_idx[int(pred_label)]] += 1.0

    total = conf.sum()
    if total <= 0.0:
        return accuracy, 0.0
    po = float(np.trace(conf) / total)
    row_marginals = conf.sum(axis=1)
    col_marginals = conf.sum(axis=0)
    pe = float(np.dot(row_marginals, col_marginals) / (total * total))
    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        kappa = 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    else:
        kappa = float((po - pe) / denom)
    return accuracy, kappa


def evaluate_batched(
    covs: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    alpha: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    preds: list[torch.Tensor] = []
    total_loss = 0.0
    total_count = 0
    for start in range(0, int(labels.numel()), batch_size):
        batch_covs = covs[start : start + batch_size].to(device=device, dtype=DEFAULT_DTYPE)
        batch_labels = labels[start : start + batch_size].to(device=device)
        scores = batch_proto_scores(batch_covs, prototypes, pairwise_fn=spd_distance_sq, pairwise_kwargs=None, alpha=alpha)
        loss = torch.nn.functional.cross_entropy(scores, batch_labels)
        preds.append(scores.argmax(dim=1).cpu())
        total_loss += float(loss.item()) * int(batch_labels.numel())
        total_count += int(batch_labels.numel())
    pred_tensor = torch.cat(preds, dim=0)
    acc, kappa = _metric_from_predictions(labels, pred_tensor)
    return {
        "accuracy": acc,
        "kappa": kappa,
        "loss": total_loss / max(total_count, 1),
    }


def safe_retraction_exp(
    x: torch.Tensor,
    xi: torch.Tensor,
    lr: float,
    *,
    eps: float = 1e-8,
    max_cond: float = 1e8,
    max_backtracks: int = 8,
) -> torch.Tensor:
    eta = float(lr)
    last_candidate = x
    for _ in range(max_backtracks + 1):
        try:
            candidate = project_spd(retraction_exp(x, xi, eta), eps=eps)
            eigvals = torch.linalg.eigvalsh(sym(candidate))
            cond = float((eigvals.max() / eigvals.min().clamp_min(eps)).item())
            if torch.isfinite(candidate).all() and cond <= max_cond:
                return candidate.detach()
            last_candidate = candidate
        except Exception:
            pass
        eta *= 0.5
    return project_spd(last_candidate, eps=eps).detach()


def aggregate_spd_metrics(prototypes: torch.Tensor, egrads: torch.Tensor, xis: torch.Tensor, norm_type: str) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for proto, egrad, xi in zip(prototypes, egrads, xis):
        try:
            rows.append(compute_metrics_spd(proto, egrad, xi, norm_type=norm_type))
        except Exception:
            continue
    if not rows:
        return {
            "rgrad_norm": float("nan"),
            "dual_norm_H": float("nan"),
            "Z_norm_sq": float("nan"),
        }
    return {
        "rgrad_norm": float(np.mean([row["rgrad_norm"] for row in rows])),
        "dual_norm_H": float(np.mean([row["dual_norm_H"] for row in rows])),
        "Z_norm_sq": float(np.mean([row["Z_norm_sq"] for row in rows])),
    }


def train_iterative(
    split: SplitData,
    *,
    class_names: list[str],
    method: str,
    lr: float,
    batch_size: int,
    max_epochs: int,
    alpha: float,
    reg_lambda: float,
    init_method: str,
    data_seed: int,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    cfg = ALL_ITERATIVE_METHODS[method]
    step_fn: StepFn = cfg["step_fn"]
    norm_type = str(cfg["norm"])

    init_proto_dict = build_initial_prototypes(
        split.train_covs.to(dtype=DEFAULT_DTYPE),
        split.train_labels,
        class_names,
        init_method=init_method,
    )
    prototypes = torch.stack([init_proto_dict[name] for name in class_names]).to(device=device, dtype=DEFAULT_DTYPE)
    init_prototypes = prototypes.detach().clone()

    dataset = TensorDataset(split.train_covs, split.train_labels)
    loader_generator = torch.Generator().manual_seed(data_seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)
    steps_per_epoch = len(loader)

    epoch_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    cumulative_step = 0
    for epoch in range(1, max_epochs + 1):
        total_loss = 0.0
        total_cls = 0.0
        total_anchor = 0.0
        total_count = 0
        last_metrics = {"rgrad_norm": float("nan"), "dual_norm_H": float("nan"), "Z_norm_sq": float("nan")}
        last_egrads: torch.Tensor | None = None
        last_xis: torch.Tensor | None = None
        for cov_batch, label_batch in loader:
            cov_batch = cov_batch.to(device=device, dtype=DEFAULT_DTYPE)
            label_batch = label_batch.to(device=device)
            proto_var = prototypes.detach().clone().requires_grad_(True)
            loss, aux = joint_proto_ce_loss(
                cov_batch,
                label_batch,
                proto_var,
                pairwise_fn=spd_distance_sq,
                pairwise_kwargs=None,
                alpha=alpha,
                reg_lambda=reg_lambda,
                init_prototypes=init_prototypes,
            )
            loss.backward()
            egrads = proto_var.grad.detach()
            xis = torch.zeros_like(prototypes)
            for class_idx in range(prototypes.shape[0]):
                xis[class_idx] = step_fn(prototypes[class_idx], egrads[class_idx])
            last_egrads = egrads
            last_xis = xis
            updated = torch.empty_like(prototypes)
            for class_idx in range(prototypes.shape[0]):
                updated[class_idx] = safe_retraction_exp(prototypes[class_idx], xis[class_idx], lr)
            prototypes = updated.detach()
            count = int(label_batch.numel())
            total_loss += float(loss.item()) * count
            total_cls += float(aux["cls_loss"].item()) * count
            total_anchor += float(aux["anchor_loss"].item()) * count
            total_count += count
            cumulative_step += 1
        if last_egrads is not None and last_xis is not None:
            last_metrics = aggregate_spd_metrics(prototypes, last_egrads, last_xis, norm_type)
        val_metrics = evaluate_batched(split.val_covs, split.val_labels, prototypes, alpha=alpha, batch_size=eval_batch_size, device=device)
        test_metrics = evaluate_batched(split.test_covs, split.test_labels, prototypes, alpha=alpha, batch_size=eval_batch_size, device=device)
        epoch_rows.append(
            {
                "epoch": epoch,
                "steps_per_epoch": steps_per_epoch,
                "cumulative_step": cumulative_step,
                "train_loss": total_loss / max(total_count, 1),
                "train_cls_loss": total_cls / max(total_count, 1),
                "train_anchor_loss": total_anchor / max(total_count, 1),
                "val_accuracy": val_metrics["accuracy"],
                "val_kappa": val_metrics["kappa"],
                "val_loss": val_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_kappa": test_metrics["kappa"],
                "test_loss": test_metrics["loss"],
                "rgrad_norm": last_metrics["rgrad_norm"],
                "dual_norm_H": last_metrics["dual_norm_H"],
                "Z_norm_sq": last_metrics["Z_norm_sq"],
                "wall_time_sec": time.perf_counter() - t0,
            }
        )
    summary = {
        "method": method,
        "lr": lr,
        "epochs": max_epochs,
        "final_val_accuracy": epoch_rows[-1]["val_accuracy"],
        "final_val_kappa": epoch_rows[-1]["val_kappa"],
        "final_val_loss": epoch_rows[-1]["val_loss"],
        "final_test_accuracy": epoch_rows[-1]["test_accuracy"],
        "final_test_kappa": epoch_rows[-1]["test_kappa"],
        "final_test_loss": epoch_rows[-1]["test_loss"],
    }
    return prototypes.detach().cpu(), epoch_rows, summary


def save_shared_artifacts(shared_dir: Path, split: SplitData, class_names: list[str], init_method: str) -> None:
    ensure_dir(shared_dir)
    torch.save(
        {"train_indices": split.train_indices, "val_indices": split.val_indices},
        shared_dir / "split.pt",
    )
    init_proto_dict = build_initial_prototypes(
        split.train_covs.to(dtype=DEFAULT_DTYPE),
        split.train_labels,
        class_names,
        init_method=init_method,
    )
    torch.save(
        {name: proto.detach().cpu() for name, proto in init_proto_dict.items()},
        shared_dir / f"init_{init_method}.pt",
    )


def choose_best_lr(rows: list[dict[str, Any]], method: str) -> float:
    method_rows = [row for row in rows if row["method"] == method]
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in method_rows:
        grouped.setdefault(float(row["lr"]), []).append(row)
    best_lr = None
    best_key = None
    for lr, lr_rows in grouped.items():
        mean_val_acc = float(np.mean([row["final_val_accuracy"] for row in lr_rows]))
        mean_val_loss = float(np.mean([row["final_val_loss"] for row in lr_rows]))
        key = (mean_val_acc, -mean_val_loss, -lr)
        if best_key is None or key > best_key:
            best_key = key
            best_lr = lr
    if best_lr is None:
        raise RuntimeError(f"no rows for method={method}")
    return best_lr


def aggregate_selected(
    rows: list[dict[str, Any]],
    methods: list[str],
    fixed_lrs: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    fixed_lrs = fixed_lrs or {}
    best_lrs = {
        method: float(fixed_lrs[method]) if method in fixed_lrs else choose_best_lr(rows, method)
        for method in methods
    }
    selected = [row for row in rows if math.isclose(float(row["lr"]), float(best_lrs[row["method"]]), rel_tol=0.0, abs_tol=1e-12)]
    mean_table = []
    for method in methods:
        method_rows = [row for row in selected if row["method"] == method]
        mean_table.append(
            {
                "method": method,
                "selected_global_lr": best_lrs[method],
                "n_seeds": len(method_rows),
                "mean_final_val_accuracy": float(np.mean([row["final_val_accuracy"] for row in method_rows])),
                "mean_final_test_accuracy": float(np.mean([row["final_test_accuracy"] for row in method_rows])),
                "std_final_test_accuracy": float(np.std([row["final_test_accuracy"] for row in method_rows], ddof=0)),
                "mean_final_test_kappa": float(np.mean([row["final_test_kappa"] for row in method_rows])),
                "std_final_test_kappa": float(np.std([row["final_test_kappa"] for row in method_rows], ddof=0)),
                "mean_final_test_loss": float(np.mean([row["final_test_loss"] for row in method_rows])),
            }
        )
    return selected, mean_table, best_lrs


def aggregate_step_curves(
    selected_rows: list[dict[str, Any]],
    methods: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        metrics_path = Path(str(row["metrics_path"]))
        with metrics_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for metric_row in reader:
                metric_rows.append(
                    {
                        "seed": int(row["seed"]),
                        "method": row["method"],
                        "lr": float(row["lr"]),
                        "epoch": int(metric_row["epoch"]),
                        "steps_per_epoch": int(metric_row["steps_per_epoch"]),
                        "cumulative_step": int(metric_row["cumulative_step"]),
                        "train_loss": float(metric_row["train_loss"]),
                        "val_accuracy": float(metric_row["val_accuracy"]),
                        "val_loss": float(metric_row["val_loss"]),
                        "test_accuracy": float(metric_row["test_accuracy"]),
                        "test_kappa": float(metric_row["test_kappa"]),
                        "test_loss": float(metric_row["test_loss"]),
                        "wall_time_sec": float(metric_row["wall_time_sec"]),
                    }
                )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault((str(row["method"]), int(row["cumulative_step"])), []).append(row)

    step_summary: list[dict[str, Any]] = []
    for method in methods:
        method_steps = sorted([step for m, step in grouped if m == method])
        for step in method_steps:
            rows = grouped[(method, step)]
            step_summary.append(
                {
                    "method": method,
                    "cumulative_step": step,
                    "n_seeds": len(rows),
                    "mean_epoch": float(np.mean([row["epoch"] for row in rows])),
                    "mean_train_loss": float(np.mean([row["train_loss"] for row in rows])),
                    "mean_val_accuracy": float(np.mean([row["val_accuracy"] for row in rows])),
                    "mean_val_loss": float(np.mean([row["val_loss"] for row in rows])),
                    "mean_test_accuracy": float(np.mean([row["test_accuracy"] for row in rows])),
                    "std_test_accuracy": float(np.std([row["test_accuracy"] for row in rows], ddof=0)),
                    "mean_test_kappa": float(np.mean([row["test_kappa"] for row in rows])),
                    "mean_test_loss": float(np.mean([row["test_loss"] for row in rows])),
                    "mean_wall_time_sec": float(np.mean([row["wall_time_sec"] for row in rows])),
                }
            )

    summary_lookup = {(row["method"], int(row["cumulative_step"])): row for row in step_summary}
    pair_defs = [
        ("main_gap_rgd_minus_egd", "riemannian_gd", "euclidean_gd"),
        ("intrinsic_gap_spd_muon_minus_rgd", "spd_muon", "riemannian_gd"),
        ("euclidean_gap_emuon_minus_egd", "euclidean_muon", "euclidean_gd"),
    ]
    all_steps = sorted({int(row["cumulative_step"]) for row in step_summary})
    pairwise_gap_rows: list[dict[str, Any]] = []
    for step in all_steps:
        for gap_name, lhs, rhs in pair_defs:
            lhs_row = summary_lookup.get((lhs, step))
            rhs_row = summary_lookup.get((rhs, step))
            if lhs_row is None or rhs_row is None:
                continue
            pairwise_gap_rows.append(
                {
                    "gap_name": gap_name,
                    "lhs_method": lhs,
                    "rhs_method": rhs,
                    "cumulative_step": step,
                    "lhs_mean_test_accuracy": lhs_row["mean_test_accuracy"],
                    "rhs_mean_test_accuracy": rhs_row["mean_test_accuracy"],
                    "test_accuracy_gap": lhs_row["mean_test_accuracy"] - rhs_row["mean_test_accuracy"],
                    "lhs_mean_test_loss": lhs_row["mean_test_loss"],
                    "rhs_mean_test_loss": rhs_row["mean_test_loss"],
                    "test_loss_gap": lhs_row["mean_test_loss"] - rhs_row["mean_test_loss"],
                }
            )
    return step_summary, pairwise_gap_rows


def parse_fixed_lrs(entries: list[str]) -> dict[str, float]:
    fixed: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"invalid --fixed-lr entry {entry!r}; expected method=lr")
        method, lr_str = entry.split("=", 1)
        method = method.strip()
        if method not in ALL_ITERATIVE_METHODS:
            raise ValueError(f"unknown method in --fixed-lr: {method!r}")
        fixed[method] = float(lr_str)
    return fixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-100 SPD prototype task from frozen spatial covariances.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=repo_root / "data" / "cifar100")
    parser.add_argument("--feature-cache-dir", type=Path, default=repo_root / "data" / "cifar100" / "features")
    parser.add_argument("--results-dir", type=Path, default=repo_root / "results" / "cifar100_spd_proto_v1")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--feature-layer", type=str, default="layer3", choices=["layer3", "layer4"])
    parser.add_argument("--label-granularity", type=str, default="coarse", choices=["coarse", "fine"])
    parser.add_argument("--projected-dim", type=int, default=32)
    parser.add_argument("--projection-seed", type=int, default=17)
    parser.add_argument("--cov-shrinkage", type=float, default=0.1)
    parser.add_argument("--cov-eps", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--lr-grid", type=float, nargs="+", default=DEFAULT_LR_GRID)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--reg-lambda", type=float, default=1e-3)
    parser.add_argument("--init-method", type=str, default="log_euclidean", choices=["arithmetic", "log_euclidean"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-per-class", type=int, default=24)
    parser.add_argument("--smoke-test-per-class", type=int, default=12)
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--fixed-lr", type=str, nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for method in args.methods:
        if method not in ALL_ITERATIVE_METHODS:
            raise ValueError(f"unknown method {method!r}")
    if args.smoke:
        args.seeds = [0]
        args.lr_grid = SMOKE_LR_GRID
        args.max_epochs = min(args.max_epochs, 2)
    fixed_lrs = parse_fixed_lrs(args.fixed_lr)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    setup_seed(0)
    data = load_or_extract_covariances(
        data_root=args.data_root,
        cache_dir=args.feature_cache_dir,
        backbone=args.backbone,
        feature_layer=args.feature_layer,
        label_granularity=args.label_granularity,
        projected_dim=args.projected_dim,
        projection_seed=args.projection_seed,
        extraction_batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        shrinkage=args.cov_shrinkage,
        eps=args.cov_eps,
        device=device,
        smoke=args.smoke,
        smoke_train_per_class=args.smoke_train_per_class,
        smoke_test_per_class=args.smoke_test_per_class,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )

    ensure_dir(args.results_dir)
    save_json(
        args.results_dir / "run_config.json",
        {
            "backbone": args.backbone,
            "feature_layer": args.feature_layer,
            "label_granularity": args.label_granularity,
            "projected_dim": args.projected_dim,
            "projection_seed": args.projection_seed,
            "cov_shrinkage": args.cov_shrinkage,
            "cov_eps": args.cov_eps,
            "seeds": args.seeds,
            "validation_fraction": args.validation_fraction,
            "methods": args.methods,
            "lr_grid": args.lr_grid,
            "max_epochs": args.max_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "feature_batch_size": args.feature_batch_size,
            "alpha": args.alpha,
            "reg_lambda": args.reg_lambda,
            "init_method": args.init_method,
            "data_source": data.source,
            "hardware": hardware_info(),
            "smoke": bool(args.smoke),
            "train_per_class": args.train_per_class,
            "test_per_class": args.test_per_class,
            "fixed_lrs": fixed_lrs,
        },
    )

    lr_sweep_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_dir = ensure_dir(args.results_dir / f"seed{seed}")
        split = split_train_val(data, seed=seed, validation_fraction=args.validation_fraction)
        save_shared_artifacts(ensure_dir(seed_dir / "shared"), split, data.class_names, args.init_method)
        for method in args.methods:
            method_dir = ensure_dir(seed_dir / method)
            method_lrs = [fixed_lrs[method]] if method in fixed_lrs else list(args.lr_grid)
            for lr in method_lrs:
                lr_dir = ensure_dir(method_dir / f"lr_{format_lr(lr)}")
                setup_seed(seed)
                prototypes, epoch_rows, summary = train_iterative(
                    split,
                    class_names=data.class_names,
                    method=method,
                    lr=float(lr),
                    batch_size=args.train_batch_size,
                    max_epochs=args.max_epochs,
                    alpha=args.alpha,
                    reg_lambda=args.reg_lambda,
                    init_method=args.init_method,
                    data_seed=seed * 1000 + int(round(float(lr) * 1e6)),
                    device=device,
                    eval_batch_size=args.eval_batch_size,
                )
                torch.save({"class_names": data.class_names, "prototypes": prototypes}, lr_dir / "class_prototypes.pt")
                write_csv(lr_dir / "metrics.csv", epoch_rows)
                save_json(
                    lr_dir / "summary.json",
                    {
                        **summary,
                        "seed": seed,
                        "method": method,
                        "lr": lr,
                    },
                )
                lr_sweep_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "lr": float(lr),
                        "final_val_accuracy": summary["final_val_accuracy"],
                        "final_val_loss": summary["final_val_loss"],
                        "final_test_accuracy": summary["final_test_accuracy"],
                        "final_test_kappa": summary["final_test_kappa"],
                        "final_test_loss": summary["final_test_loss"],
                        "wall_time_sec": epoch_rows[-1]["wall_time_sec"],
                        "metrics_path": str(lr_dir / "metrics.csv"),
                    }
                )

    write_csv(args.results_dir / "lr_sweep_summary.csv", lr_sweep_rows)
    selected_rows, mean_table, best_lrs = aggregate_selected(lr_sweep_rows, args.methods, fixed_lrs=fixed_lrs)
    write_csv(args.results_dir / "global_selected_summary.csv", selected_rows)
    write_csv(args.results_dir / "mean_test_accuracy_table.csv", mean_table)
    save_json(args.results_dir / "global_best_lr.json", best_lrs)
    step_summary, pairwise_gap_rows = aggregate_step_curves(selected_rows, args.methods)
    write_csv(args.results_dir / "step_summary.csv", step_summary)
    write_csv(args.results_dir / "pairwise_gap_summary.csv", pairwise_gap_rows)


if __name__ == "__main__":
    main()
