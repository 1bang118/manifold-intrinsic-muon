from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models

from src.utils import DEFAULT_DTYPE, ensure_dir, format_lr, hardware_info, matrix_power_symmetric, ortho, save_json, setup_seed, write_csv


torch.set_default_dtype(DEFAULT_DTYPE)


DEFAULT_LR_GRID = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
SMOKE_LR_GRID = [3e-2, 1e-1]
DEFAULT_RANKS = [10, 20, 40]
SMOKE_RANKS = [8]
DEFAULT_METHODS = [
    "riemannian_gd",
    "scaled_muon",
    "scaled_numuon",
    "euclidean_gd",
    "euclidean_muon",
    "euclidean_numuon",
]


@dataclass(frozen=True)
class FeatureData:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    n_classes: int
    feature_dim: int
    source: str


@dataclass(frozen=True)
class SplitData:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    split_indices: dict[str, np.ndarray]


@dataclass(frozen=True)
class HeadState:
    left: torch.Tensor
    right: torch.Tensor
    bias: torch.Tensor


def _backbone_spec(name: str) -> tuple[Any, Any]:
    if name == "resnet18":
        return models.resnet18, models.ResNet18_Weights.DEFAULT
    if name == "resnet50":
        return models.resnet50, models.ResNet50_Weights.DEFAULT
    raise ValueError(f"Unsupported backbone: {name}")


def _load_or_extract_cifar100_features(
    *,
    root: Path,
    cache_dir: Path,
    backbone: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    l2_normalize: bool,
) -> FeatureData:
    cache_path = cache_dir / f"cifar100_{backbone}_{'l2' if l2_normalize else 'raw'}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return FeatureData(
            train_features=payload["train_features"].to(dtype=DEFAULT_DTYPE),
            train_labels=payload["train_labels"].long(),
            test_features=payload["test_features"].to(dtype=DEFAULT_DTYPE),
            test_labels=payload["test_labels"].long(),
            n_classes=int(payload["n_classes"]),
            feature_dim=int(payload["feature_dim"]),
            source=str(cache_path),
        )

    model_fn, weights = _backbone_spec(backbone)
    transform = weights.transforms()
    train_ds = datasets.CIFAR100(root=str(root), train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100(root=str(root), train=False, download=True, transform=transform)

    backbone_model = model_fn(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(backbone_model.children())[:-1], torch.nn.Flatten()).to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad_(False)

    def extract_split(ds: torch.utils.data.Dataset[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        feats: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        with torch.no_grad():
            for images, target in loader:
                images = images.to(device=device, dtype=DEFAULT_DTYPE)
                feature_batch = feature_extractor(images)
                if l2_normalize:
                    feature_batch = F.normalize(feature_batch, p=2, dim=1)
                feats.append(feature_batch.cpu())
                labels.append(target.cpu())
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

    train_features, train_labels = extract_split(train_ds)
    test_features, test_labels = extract_split(test_ds)
    ensure_dir(cache_dir)
    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels,
            "n_classes": 100,
            "feature_dim": int(train_features.shape[1]),
            "backbone": backbone,
            "l2_normalize": l2_normalize,
        },
        cache_path,
    )
    return FeatureData(
        train_features=train_features.to(dtype=DEFAULT_DTYPE),
        train_labels=train_labels.long(),
        test_features=test_features.to(dtype=DEFAULT_DTYPE),
        test_labels=test_labels.long(),
        n_classes=100,
        feature_dim=int(train_features.shape[1]),
        source=str(cache_path),
    )


def _make_synthetic_features(
    *,
    seed: int,
    n_train: int = 1200,
    n_test: int = 600,
    feature_dim: int = 128,
    n_classes: int = 20,
) -> FeatureData:
    rng = np.random.default_rng(seed)
    class_basis = rng.normal(size=(n_classes, feature_dim))
    class_basis /= np.linalg.norm(class_basis, axis=1, keepdims=True).clip(min=1e-12)
    confusion_basis = rng.normal(scale=0.35, size=(max(1, n_classes // 4), feature_dim))

    def sample_split(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, n_classes, size=n_samples)
        features = np.zeros((n_samples, feature_dim), dtype=np.float64)
        for i, label in enumerate(labels):
            group = label % confusion_basis.shape[0]
            features[i] = (
                1.6 * class_basis[label]
                + 0.9 * confusion_basis[group]
                + 0.25 * class_basis[(label + 1) % n_classes]
                + rng.normal(scale=0.35, size=feature_dim)
            )
        norms = np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)
        features = features / norms
        return features, labels

    train_features, train_labels = sample_split(n_train)
    test_features, test_labels = sample_split(n_test)
    return FeatureData(
        train_features=torch.as_tensor(train_features, dtype=DEFAULT_DTYPE),
        train_labels=torch.as_tensor(train_labels, dtype=torch.long),
        test_features=torch.as_tensor(test_features, dtype=DEFAULT_DTYPE),
        test_labels=torch.as_tensor(test_labels, dtype=torch.long),
        n_classes=n_classes,
        feature_dim=feature_dim,
        source="synthetic",
    )


def load_feature_data(args: argparse.Namespace, device: torch.device) -> FeatureData:
    if args.synthetic:
        return _make_synthetic_features(seed=args.seed)
    data_root = Path(args.data_root)
    cache_dir = Path(args.feature_cache_dir)
    return _load_or_extract_cifar100_features(
        root=data_root,
        cache_dir=cache_dir,
        backbone=args.backbone,
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        l2_normalize=args.feature_l2_normalize,
    )


def split_features(
    data: FeatureData,
    *,
    seed: int,
    validation_fraction: float,
    device: torch.device,
) -> SplitData:
    rng = np.random.default_rng(seed)
    n_train = int(data.train_labels.numel())
    perm = rng.permutation(n_train)
    n_val = max(data.n_classes, int(round(validation_fraction * n_train)))
    n_val = min(n_val, n_train - data.n_classes)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    return SplitData(
        train_features=data.train_features[train_idx].to(device=device, dtype=DEFAULT_DTYPE),
        train_labels=data.train_labels[train_idx].to(device=device),
        val_features=data.train_features[val_idx].to(device=device, dtype=DEFAULT_DTYPE),
        val_labels=data.train_labels[val_idx].to(device=device),
        test_features=data.test_features.to(device=device, dtype=DEFAULT_DTYPE),
        test_labels=data.test_labels.to(device=device),
        split_indices={"train": train_idx, "val": val_idx},
    )


def init_head(
    feature_dim: int,
    n_classes: int,
    rank: int,
    seed: int,
    device: torch.device,
    *,
    gauge_alpha: float = 1.0,
    init_mode: str = "random",
    split: SplitData | None = None,
) -> HeadState:
    if init_mode == "random":
        g = torch.Generator(device=device).manual_seed(seed)
        scale = 1.0 / math.sqrt(rank)
        left = torch.randn((feature_dim, rank), generator=g, device=device, dtype=DEFAULT_DTYPE) * scale
        right = torch.randn((rank, n_classes), generator=g, device=device, dtype=DEFAULT_DTYPE) * scale
    elif init_mode == "class_mean_svd":
        if split is None:
            raise ValueError("class_mean_svd initialization requires split")
        means = torch.zeros((feature_dim, n_classes), dtype=DEFAULT_DTYPE, device=device)
        counts = torch.zeros((n_classes,), dtype=DEFAULT_DTYPE, device=device)
        means.index_add_(1, split.train_labels, split.train_features.transpose(0, 1))
        counts.index_add_(0, split.train_labels, torch.ones_like(split.train_labels, dtype=DEFAULT_DTYPE))
        means = means / counts.clamp_min(1.0).unsqueeze(0)
        u, s, vh = torch.linalg.svd(means, full_matrices=False)
        k = min(rank, int(s.numel()))
        sqrt_s = torch.sqrt(s[:k].clamp_min(0.0))
        left = torch.zeros((feature_dim, rank), dtype=DEFAULT_DTYPE, device=device)
        right = torch.zeros((rank, n_classes), dtype=DEFAULT_DTYPE, device=device)
        left[:, :k] = u[:, :k] * sqrt_s.unsqueeze(0)
        right[:k, :] = sqrt_s.unsqueeze(1) * vh[:k, :]
        if k < rank:
            g = torch.Generator(device=device).manual_seed(seed + 80_003)
            left[:, k:] = 1e-4 * torch.randn((feature_dim, rank - k), generator=g, device=device, dtype=DEFAULT_DTYPE)
            right[k:, :] = 1e-4 * torch.randn((rank - k, n_classes), generator=g, device=device, dtype=DEFAULT_DTYPE)
    else:
        raise ValueError(f"unknown init_mode={init_mode!r}")
    alpha = float(gauge_alpha)
    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError(f"gauge_alpha must be a finite positive value, got {gauge_alpha}")
    left = left * alpha
    right = right / alpha
    bias = torch.zeros((n_classes,), device=device, dtype=DEFAULT_DTYPE)
    return HeadState(left=left, right=right, bias=bias)


def save_shared_artifacts(shared_dir: Path, rank: int, seed: int, split: SplitData, init_state: HeadState) -> None:
    ensure_dir(shared_dir)
    torch.save(
        {
            "train_indices": split.split_indices["train"],
            "val_indices": split.split_indices["val"],
        },
        shared_dir / f"split_seed{seed}_rank{rank}.pt",
    )
    torch.save(
        {
            "left": init_state.left.detach().cpu(),
            "right": init_state.right.detach().cpu(),
            "bias": init_state.bias.detach().cpu(),
        },
        shared_dir / f"init_seed{seed}_rank{rank}.pt",
    )


def factor_grams(state: HeadState, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    s_left = state.left.transpose(-1, -2) @ state.left
    s_right = state.right @ state.right.transpose(-1, -2)
    eye = torch.eye(s_left.shape[-1], dtype=state.left.dtype, device=state.left.device)
    return s_left + eps * eye, s_right + eps * eye


def safe_invsqrt_spd(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return matrix_power_symmetric(x + eps * eye, -0.5, eps=eps)


def logits_from_state(state: HeadState, features: torch.Tensor) -> torch.Tensor:
    return features @ (state.left @ state.right) + state.bias


def objective_loss(
    state: HeadState,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight_decay: float,
    bias_weight_decay: float,
) -> torch.Tensor:
    logits = logits_from_state(state, features)
    ce = F.cross_entropy(logits, labels)
    w = state.left @ state.right
    reg = 0.5 * weight_decay * torch.sum(w * w) + 0.5 * bias_weight_decay * torch.sum(state.bias * state.bias)
    return ce + reg


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).to(dtype=DEFAULT_DTYPE).mean().item())


def evaluate_split(
    state: HeadState,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight_decay: float,
    bias_weight_decay: float,
    batch_size: int,
) -> dict[str, float]:
    losses: list[float] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, int(labels.numel()), batch_size):
            end = min(start + batch_size, int(labels.numel()))
            f_batch = features[start:end]
            y_batch = labels[start:end]
            logits = logits_from_state(state, f_batch)
            ce = F.cross_entropy(logits, y_batch, reduction="sum")
            losses.append(float(ce.item()))
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == y_batch).sum().item())
            total += int(y_batch.numel())
    ce_loss = float(sum(losses) / max(total, 1))
    reg = 0.5 * weight_decay * float(torch.sum((state.left @ state.right) ** 2).item())
    reg += 0.5 * bias_weight_decay * float(torch.sum(state.bias ** 2).item())
    return {
        "loss": ce_loss + reg,
        "ce_loss": ce_loss,
        "acc": float(correct / max(total, 1)),
    }


def condition_number(x: torch.Tensor, eps: float = 1e-12) -> float:
    s = torch.linalg.svdvals(x)
    if s.numel() == 0:
        return 0.0
    return float((s.max() / s.min().clamp_min(eps)).item())


def factor_norm_sq(state: HeadState) -> float:
    return float((state.left.pow(2).sum() + state.right.pow(2).sum()).item())


def pair_fro_norm(x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(x_left * x_left) + torch.sum(x_right * x_right))


def pair_dual_norm(h_left: torch.Tensor, h_right: torch.Tensor, norm_type: str) -> float:
    if norm_type == "frobenius":
        return float(pair_fro_norm(h_left, h_right).item())
    if norm_type == "frobenius_block":
        return float(torch.linalg.norm(h_left, ord="fro").item() + torch.linalg.norm(h_right, ord="fro").item())
    if norm_type == "spectral":
        return float(torch.linalg.norm(h_left, ord="nuc").item() + torch.linalg.norm(h_right, ord="nuc").item())
    if norm_type == "nuclear":
        return float(max(torch.linalg.norm(h_left, ord=2).item(), torch.linalg.norm(h_right, ord=2).item()))
    raise ValueError(norm_type)


def whitened_gradients(
    state: HeadState,
    egrad_left: torch.Tensor,
    egrad_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_left, s_right = factor_grams(state)
    invsqrt_left = safe_invsqrt_spd(s_left)
    invsqrt_right = safe_invsqrt_spd(s_right)
    h_left = egrad_left @ invsqrt_right
    h_right = invsqrt_left @ egrad_right
    return h_left, h_right, invsqrt_left, invsqrt_right


def riemannian_gd_step(state: HeadState, egrad_left: torch.Tensor, egrad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h_left, h_right, invsqrt_left, invsqrt_right = whitened_gradients(state, egrad_left, egrad_right)
    denom = pair_fro_norm(h_left, h_right)
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-14:
        return torch.zeros_like(state.left), torch.zeros_like(state.right)
    z_left = h_left / denom
    z_right = h_right / denom
    return z_left @ invsqrt_right, invsqrt_left @ z_right


def riemannian_gd_block_step(state: HeadState, egrad_left: torch.Tensor, egrad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h_left, h_right, invsqrt_left, invsqrt_right = whitened_gradients(state, egrad_left, egrad_right)
    denom_left = torch.linalg.norm(h_left, ord="fro")
    denom_right = torch.linalg.norm(h_right, ord="fro")
    z_left = torch.zeros_like(h_left) if (not torch.isfinite(denom_left) or float(denom_left.item()) <= 1e-14) else h_left / denom_left
    z_right = torch.zeros_like(h_right) if (not torch.isfinite(denom_right) or float(denom_right.item()) <= 1e-14) else h_right / denom_right
    return z_left @ invsqrt_right, invsqrt_left @ z_right


def scaled_muon_step(state: HeadState, egrad_left: torch.Tensor, egrad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h_left, h_right, invsqrt_left, invsqrt_right = whitened_gradients(state, egrad_left, egrad_right)
    if float(pair_fro_norm(h_left, h_right).item()) <= 1e-14:
        return torch.zeros_like(state.left), torch.zeros_like(state.right)
    z_left = ortho(h_left)
    z_right = ortho(h_right.transpose(-1, -2)).transpose(-1, -2)
    return z_left @ invsqrt_right, invsqrt_left @ z_right


def _rank1(x: torch.Tensor) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    if s.numel() == 0 or float(s[0].item()) <= 1e-14:
        return torch.zeros_like(x)
    return u[:, :1] @ vh[:1, :]


def scaled_numuon_step(state: HeadState, egrad_left: torch.Tensor, egrad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h_left, h_right, invsqrt_left, invsqrt_right = whitened_gradients(state, egrad_left, egrad_right)
    if max(float(torch.linalg.norm(h_left, ord=2).item()), float(torch.linalg.norm(h_right, ord=2).item())) <= 1e-14:
        return torch.zeros_like(state.left), torch.zeros_like(state.right)
    z_left = _rank1(h_left)
    z_right = _rank1(h_right)
    return z_left @ invsqrt_right, invsqrt_left @ z_right


def euclidean_gd_step(state: HeadState, egrad_left: torch.Tensor, egrad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denom = pair_fro_norm(egrad_left, egrad_right)
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-14:
        return torch.zeros_like(state.left), torch.zeros_like(state.right)
    return egrad_left / denom, egrad_right / denom


def euclidean_gd_block_step(state: HeadState, egrad_left: torch.Tensor, egrad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denom_left = torch.linalg.norm(egrad_left, ord="fro")
    denom_right = torch.linalg.norm(egrad_right, ord="fro")
    xi_left = torch.zeros_like(state.left) if (not torch.isfinite(denom_left) or float(denom_left.item()) <= 1e-14) else egrad_left / denom_left
    xi_right = torch.zeros_like(state.right) if (not torch.isfinite(denom_right) or float(denom_right.item()) <= 1e-14) else egrad_right / denom_right
    return xi_left, xi_right


def euclidean_muon_step(state: HeadState, egrad_left: torch.Tensor, egrad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if float(pair_fro_norm(egrad_left, egrad_right).item()) <= 1e-14:
        return torch.zeros_like(state.left), torch.zeros_like(state.right)
    return ortho(egrad_left), ortho(egrad_right.transpose(-1, -2)).transpose(-1, -2)


def spectron_muon_step(state: HeadState, egrad_left: torch.Tensor, egrad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if float(pair_fro_norm(egrad_left, egrad_right).item()) <= 1e-14:
        return torch.zeros_like(state.left), torch.zeros_like(state.right)
    denom = float(torch.linalg.norm(state.left, ord=2).item() + torch.linalg.norm(state.right, ord=2).item() + 1.0)
    if denom <= 0.0 or not math.isfinite(denom):
        return torch.zeros_like(state.left), torch.zeros_like(state.right)
    left_step = ortho(egrad_left) / denom
    right_step = ortho(egrad_right.transpose(-1, -2)).transpose(-1, -2) / denom
    return left_step, right_step


def euclidean_numuon_step(state: HeadState, egrad_left: torch.Tensor, egrad_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if max(float(torch.linalg.norm(egrad_left, ord=2).item()), float(torch.linalg.norm(egrad_right, ord=2).item())) <= 1e-14:
        return torch.zeros_like(state.left), torch.zeros_like(state.right)
    return _rank1(egrad_left), _rank1(egrad_right)


METHODS: dict[str, dict[str, Any]] = {
    "riemannian_gd": {"step_fn": riemannian_gd_step, "norm": "frobenius", "geometry": "coupled"},
    "riemannian_gd_block": {"step_fn": riemannian_gd_block_step, "norm": "frobenius_block", "geometry": "coupled"},
    "scaled_muon": {"step_fn": scaled_muon_step, "norm": "spectral", "geometry": "coupled"},
    "scaled_numuon": {"step_fn": scaled_numuon_step, "norm": "nuclear", "geometry": "coupled"},
    "euclidean_gd": {"step_fn": euclidean_gd_step, "norm": "frobenius", "geometry": "euclidean"},
    "euclidean_gd_block": {"step_fn": euclidean_gd_block_step, "norm": "frobenius_block", "geometry": "euclidean"},
    "euclidean_muon": {"step_fn": euclidean_muon_step, "norm": "spectral", "geometry": "euclidean"},
    "spectron_muon": {"step_fn": spectron_muon_step, "norm": "spectral", "geometry": "euclidean"},
    "euclidean_numuon": {"step_fn": euclidean_numuon_step, "norm": "nuclear", "geometry": "euclidean"},
}


def retract_factors(
    state: HeadState,
    xi_left: torch.Tensor,
    xi_right: torch.Tensor,
    lr: float,
    *,
    min_sigma: float = 1e-8,
    max_backtracks: int = 12,
) -> HeadState:
    step = float(lr)
    last = state
    for _ in range(max_backtracks + 1):
        left_new = state.left - step * xi_left
        right_new = state.right - step * xi_right
        if not torch.isfinite(left_new).all() or not torch.isfinite(right_new).all():
            step *= 0.5
            continue
        smin_left = float(torch.linalg.svdvals(left_new).min().item())
        smin_right = float(torch.linalg.svdvals(right_new).min().item())
        cand = HeadState(left=left_new.detach(), right=right_new.detach(), bias=state.bias)
        last = cand
        if smin_left > min_sigma and smin_right > min_sigma:
            return cand
        step *= 0.5
    return last


def compute_batch_matrix_gradient(
    state: HeadState,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight_decay: float,
) -> torch.Tensor:
    with torch.no_grad():
        logits = logits_from_state(state, features)
        probs = torch.softmax(logits, dim=1)
        targets = F.one_hot(labels, num_classes=logits.shape[1]).to(dtype=DEFAULT_DTYPE)
        grad_logits = (probs - targets) / float(labels.numel())
        grad_w = features.transpose(0, 1) @ grad_logits + weight_decay * (state.left @ state.right)
        return grad_w


def spectrum_stats(grad_w: torch.Tensor) -> dict[str, float]:
    s = torch.linalg.svdvals(grad_w)
    if s.numel() == 0:
        return {"grad_sigma1": 0.0, "grad_sigma2": 0.0, "grad_sigma_ratio": 0.0, "grad_top1_energy": 0.0, "grad_top5_energy": 0.0}
    energy = torch.sum(s * s).clamp_min(1e-30)
    sigma1 = float(s[0].item())
    sigma2 = float(s[1].item()) if s.numel() > 1 else 0.0
    top1 = float((s[0] * s[0] / energy).item())
    top5 = float((torch.sum(s[: min(5, s.numel())] ** 2) / energy).item())
    return {
        "grad_sigma1": sigma1,
        "grad_sigma2": sigma2,
        "grad_sigma_ratio": sigma1 / max(sigma2, 1e-12),
        "grad_top1_energy": top1,
        "grad_top5_energy": top5,
    }


def pair_signal_stats(x_left: torch.Tensor, x_right: torch.Tensor, prefix: str) -> dict[str, float]:
    s_left = torch.linalg.svdvals(x_left)
    s_right = torch.linalg.svdvals(x_right)
    s = torch.cat([s_left, s_right])
    if s.numel() == 0:
        return {
            f"{prefix}_sigma1": 0.0,
            f"{prefix}_sigma2": 0.0,
            f"{prefix}_sigma_ratio": 0.0,
            f"{prefix}_top1_energy": 0.0,
            f"{prefix}_top5_energy": 0.0,
            f"{prefix}_effective_rank": 0.0,
        }
    s = torch.sort(s, descending=True).values
    energy = torch.sum(s * s).clamp_min(1e-30)
    sigma1 = float(s[0].item())
    sigma2 = float(s[1].item()) if s.numel() > 1 else 0.0
    top1 = float((s[0] * s[0] / energy).item())
    top5 = float((torch.sum(s[: min(5, s.numel())] ** 2) / energy).item())
    effective_rank = float((torch.sum(s).pow(2) / energy).item())
    return {
        f"{prefix}_sigma1": sigma1,
        f"{prefix}_sigma2": sigma2,
        f"{prefix}_sigma_ratio": sigma1 / max(sigma2, 1e-12),
        f"{prefix}_top1_energy": top1,
        f"{prefix}_top5_energy": top5,
        f"{prefix}_effective_rank": effective_rank,
    }


def geometry_diagnostic_metrics(
    state: HeadState,
    egrad_left: torch.Tensor,
    egrad_right: torch.Tensor,
) -> dict[str, float]:
    h_left, h_right, _, _ = whitened_gradients(state, egrad_left, egrad_right)
    raw_norm = pair_fro_norm(egrad_left, egrad_right).clamp_min(1e-30)
    white_norm = pair_fro_norm(h_left, h_right).clamp_min(1e-30)
    raw_left = egrad_left / raw_norm
    raw_right = egrad_right / raw_norm
    white_left = h_left / white_norm
    white_right = h_right / white_norm
    cosine = torch.sum(raw_left * white_left) + torch.sum(raw_right * white_right)
    cosine = float(torch.clamp(cosine, min=-1.0, max=1.0).item())
    stats = {
        "raw_vs_whitened_cosine": cosine,
        "delta_geom": 1.0 - cosine,
    }
    stats.update(pair_signal_stats(egrad_left, egrad_right, "raw_pair"))
    stats.update(pair_signal_stats(h_left, h_right, "whitened_pair"))
    return stats


def method_metrics(
    state: HeadState,
    egrad_left: torch.Tensor,
    egrad_right: torch.Tensor,
    xi_left: torch.Tensor,
    xi_right: torch.Tensor,
    norm_type: str,
    geometry: str,
) -> dict[str, float]:
    if geometry == "coupled":
        h_left, h_right, _, _ = whitened_gradients(state, egrad_left, egrad_right)
        z_left = xi_left @ matrix_power_symmetric(factor_grams(state)[1], 0.5)
        z_right = matrix_power_symmetric(factor_grams(state)[0], 0.5) @ xi_right
        rgrad_norm = float(pair_fro_norm(h_left, h_right).item())
    else:
        h_left, h_right = egrad_left, egrad_right
        z_left, z_right = xi_left, xi_right
        rgrad_norm = float(pair_fro_norm(egrad_left, egrad_right).item())
    out = {
        "rgrad_norm": rgrad_norm,
        "dual_norm_H": pair_dual_norm(h_left, h_right, norm_type),
        "Z_norm_sq": float((torch.sum(z_left * z_left) + torch.sum(z_right * z_right)).item()),
        "kappa_left": condition_number(state.left),
        "kappa_right": condition_number(state.right),
        "factor_norm_sq": factor_norm_sq(state),
    }
    out.update(geometry_diagnostic_metrics(state, egrad_left, egrad_right))
    return out


def convergence_row(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    test_loss: float,
    test_acc: float,
    wall_time_sec: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "wall_time_sec": wall_time_sec,
    }
    row.update(metrics)
    return row


def run_one_training(
    split: SplitData,
    init_state: HeadState,
    *,
    method: str,
    lr: float,
    max_epochs: int,
    batch_size: int,
    weight_decay: float,
    bias_weight_decay: float,
    bias_lr_multiplier: float,
    eval_batch_size: int,
    batch_seed_base: int,
) -> tuple[HeadState, list[dict[str, Any]], dict[str, Any]]:
    cfg = METHODS[method]
    state = HeadState(left=init_state.left.clone(), right=init_state.right.clone(), bias=init_state.bias.clone())
    logs: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    n_train = int(split.train_labels.numel())

    for epoch in range(1, max_epochs + 1):
        batch_gen = torch.Generator(device=split.train_features.device).manual_seed(batch_seed_base + epoch)
        perm = torch.randperm(n_train, generator=batch_gen, device=split.train_features.device)
        batch_metrics: list[dict[str, float]] = []
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : min(start + batch_size, n_train)]
            feat_batch = split.train_features.index_select(0, idx)
            label_batch = split.train_labels.index_select(0, idx)

            left_var = state.left.detach().clone().requires_grad_(True)
            right_var = state.right.detach().clone().requires_grad_(True)
            bias_var = state.bias.detach().clone().requires_grad_(True)
            var_state = HeadState(left=left_var, right=right_var, bias=bias_var)
            loss = objective_loss(
                var_state,
                feat_batch,
                label_batch,
                weight_decay=weight_decay,
                bias_weight_decay=bias_weight_decay,
            )
            loss.backward()

            egrad_left = left_var.grad.detach()
            egrad_right = right_var.grad.detach()
            bias_grad = bias_var.grad.detach()
            xi_left, xi_right = cfg["step_fn"](state, egrad_left, egrad_right)

            metric_row = method_metrics(state, egrad_left, egrad_right, xi_left, xi_right, cfg["norm"], cfg["geometry"])
            grad_w = compute_batch_matrix_gradient(state, feat_batch, label_batch, weight_decay=weight_decay)
            metric_row.update(spectrum_stats(grad_w))
            batch_metrics.append(metric_row)

            state = retract_factors(state, xi_left, xi_right, lr)
            bias_new = state.bias - float(lr * bias_lr_multiplier) * bias_grad
            state = HeadState(left=state.left, right=state.right, bias=bias_new.detach())

            with torch.no_grad():
                logits = logits_from_state(state, feat_batch)
                train_loss_sum += float(loss.item()) * int(label_batch.numel())
                train_correct += int((torch.argmax(logits, dim=1) == label_batch).sum().item())
                train_count += int(label_batch.numel())

        train_loss = train_loss_sum / max(train_count, 1)
        train_acc = float(train_correct / max(train_count, 1))
        val_eval = evaluate_split(
            state,
            split.val_features,
            split.val_labels,
            weight_decay=weight_decay,
            bias_weight_decay=bias_weight_decay,
            batch_size=max(eval_batch_size, batch_size),
        )
        test_eval = evaluate_split(
            state,
            split.test_features,
            split.test_labels,
            weight_decay=weight_decay,
            bias_weight_decay=bias_weight_decay,
            batch_size=max(eval_batch_size, batch_size),
        )
        elapsed = time.perf_counter() - t0
        avg_metrics = {
            key: float(np.mean([row[key] for row in batch_metrics])) for key in batch_metrics[0]
        }
        logs.append(
            convergence_row(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_eval["loss"],
                val_acc=val_eval["acc"],
                test_loss=test_eval["loss"],
                test_acc=test_eval["acc"],
                wall_time_sec=elapsed,
                metrics=avg_metrics,
            )
        )

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
        "final_rgrad_norm": final["rgrad_norm"],
        "final_dual_norm_H": final["dual_norm_H"],
        "final_Z_norm_sq": final["Z_norm_sq"],
        "final_kappa_left": final["kappa_left"],
        "final_kappa_right": final["kappa_right"],
        "final_factor_norm_sq": final["factor_norm_sq"],
        "final_grad_sigma1": final["grad_sigma1"],
        "final_grad_sigma2": final["grad_sigma2"],
        "final_grad_sigma_ratio": final["grad_sigma_ratio"],
        "final_grad_top1_energy": final["grad_top1_energy"],
        "final_grad_top5_energy": final["grad_top5_energy"],
        "final_raw_vs_whitened_cosine": final["raw_vs_whitened_cosine"],
        "final_delta_geom": final["delta_geom"],
        "final_raw_pair_sigma_ratio": final["raw_pair_sigma_ratio"],
        "final_raw_pair_top1_energy": final["raw_pair_top1_energy"],
        "final_raw_pair_top5_energy": final["raw_pair_top5_energy"],
        "final_raw_pair_effective_rank": final["raw_pair_effective_rank"],
        "final_whitened_pair_sigma_ratio": final["whitened_pair_sigma_ratio"],
        "final_whitened_pair_top1_energy": final["whitened_pair_top1_energy"],
        "final_whitened_pair_top5_energy": final["whitened_pair_top5_energy"],
        "final_whitened_pair_effective_rank": final["whitened_pair_effective_rank"],
        "final_wall_time_sec": final["wall_time_sec"],
    }
    return state, logs, summary


def save_run_outputs(
    run_dir: Path,
    config: dict[str, Any],
    init_path: Path,
    logs: list[dict[str, Any]],
    final_state: HeadState,
    summary: dict[str, Any],
) -> None:
    ensure_dir(run_dir)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "init_checkpoint.json", {"path": str(init_path)})
    write_csv(run_dir / "metrics.csv", logs)
    torch.save(
        {
            "left": final_state.left.detach().cpu(),
            "right": final_state.right.detach().cpu(),
            "bias": final_state.bias.detach().cpu(),
        },
        run_dir / "final_model.pt",
    )
    save_json(run_dir / "summary.json", summary)


def rank_sweep_summary(results_dir: Path, rank: int, methods: list[str], seeds: list[int], lr_grid: list[float]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    best: dict[str, float] = {}
    for method in methods:
        candidates = []
        for lr in lr_grid:
            val_accs = []
            val_losses = []
            test_accs = []
            for seed in seeds:
                p = results_dir / f"seed{seed}" / f"rank{rank}" / method / f"lr_{format_lr(lr)}" / "summary.json"
                if not p.exists():
                    continue
                import json

                d = json.loads(p.read_text())
                val_accs.append(float(d["final_val_acc"]))
                val_losses.append(float(d["final_val_loss"]))
                test_accs.append(float(d["final_test_acc"]))
            if val_accs:
                row = {
                    "rank": rank,
                    "method": method,
                    "lr": float(lr),
                    "n_seeds": len(val_accs),
                    "mean_final_val_acc": float(np.mean(val_accs)),
                    "std_final_val_acc": float(np.std(val_accs, ddof=1)) if len(val_accs) > 1 else 0.0,
                    "mean_final_val_loss": float(np.mean(val_losses)),
                    "std_final_val_loss": float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else 0.0,
                    "mean_final_test_acc": float(np.mean(test_accs)),
                    "std_final_test_acc": float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
                }
                rows.append(row)
                candidates.append(row)
        if candidates:
            chosen = max(candidates, key=lambda row: (row["mean_final_val_acc"], -row["mean_final_val_loss"], row["n_seeds"], -row["lr"]))
            best[method] = float(chosen["lr"])
    return rows, best


def aggregate_best_runs(results_dir: Path, rank: int, best_lrs: dict[str, float], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for method, lr in best_lrs.items():
        values = {
            "final_val_acc": [],
            "final_val_loss": [],
            "final_test_acc": [],
            "final_test_loss": [],
            "final_train_acc": [],
            "final_train_loss": [],
            "final_wall_time_sec": [],
            "final_grad_sigma_ratio": [],
            "final_grad_top1_energy": [],
            "final_grad_top5_energy": [],
            "final_raw_vs_whitened_cosine": [],
            "final_delta_geom": [],
            "final_raw_pair_sigma_ratio": [],
            "final_raw_pair_top1_energy": [],
            "final_raw_pair_top5_energy": [],
            "final_raw_pair_effective_rank": [],
            "final_whitened_pair_sigma_ratio": [],
            "final_whitened_pair_top1_energy": [],
            "final_whitened_pair_top5_energy": [],
            "final_whitened_pair_effective_rank": [],
        }
        for seed in seeds:
            p = results_dir / f"seed{seed}" / f"rank{rank}" / method / f"lr_{format_lr(lr)}" / "summary.json"
            if not p.exists():
                continue
            import json

            d = json.loads(p.read_text())
            for key in values:
                values[key].append(float(d[key]))
        rows.append(
            {
                "rank": rank,
                "method": method,
                "best_lr": lr,
                "n_seeds": len(values["final_test_acc"]),
                "mean_final_val_acc": float(np.mean(values["final_val_acc"])) if values["final_val_acc"] else None,
                "mean_final_val_loss": float(np.mean(values["final_val_loss"])) if values["final_val_loss"] else None,
                "mean_final_test_acc": float(np.mean(values["final_test_acc"])) if values["final_test_acc"] else None,
                "std_final_test_acc": float(np.std(values["final_test_acc"], ddof=1)) if len(values["final_test_acc"]) > 1 else 0.0 if values["final_test_acc"] else None,
                "mean_final_test_loss": float(np.mean(values["final_test_loss"])) if values["final_test_loss"] else None,
                "mean_final_train_acc": float(np.mean(values["final_train_acc"])) if values["final_train_acc"] else None,
                "mean_final_train_loss": float(np.mean(values["final_train_loss"])) if values["final_train_loss"] else None,
                "mean_final_wall_time_sec": float(np.mean(values["final_wall_time_sec"])) if values["final_wall_time_sec"] else None,
                "mean_final_grad_sigma_ratio": float(np.mean(values["final_grad_sigma_ratio"])) if values["final_grad_sigma_ratio"] else None,
                "mean_final_grad_top1_energy": float(np.mean(values["final_grad_top1_energy"])) if values["final_grad_top1_energy"] else None,
                "mean_final_grad_top5_energy": float(np.mean(values["final_grad_top5_energy"])) if values["final_grad_top5_energy"] else None,
                "mean_final_raw_vs_whitened_cosine": float(np.mean(values["final_raw_vs_whitened_cosine"])) if values["final_raw_vs_whitened_cosine"] else None,
                "mean_final_delta_geom": float(np.mean(values["final_delta_geom"])) if values["final_delta_geom"] else None,
                "mean_final_raw_pair_sigma_ratio": float(np.mean(values["final_raw_pair_sigma_ratio"])) if values["final_raw_pair_sigma_ratio"] else None,
                "mean_final_raw_pair_top1_energy": float(np.mean(values["final_raw_pair_top1_energy"])) if values["final_raw_pair_top1_energy"] else None,
                "mean_final_raw_pair_top5_energy": float(np.mean(values["final_raw_pair_top5_energy"])) if values["final_raw_pair_top5_energy"] else None,
                "mean_final_raw_pair_effective_rank": float(np.mean(values["final_raw_pair_effective_rank"])) if values["final_raw_pair_effective_rank"] else None,
                "mean_final_whitened_pair_sigma_ratio": float(np.mean(values["final_whitened_pair_sigma_ratio"])) if values["final_whitened_pair_sigma_ratio"] else None,
                "mean_final_whitened_pair_top1_energy": float(np.mean(values["final_whitened_pair_top1_energy"])) if values["final_whitened_pair_top1_energy"] else None,
                "mean_final_whitened_pair_top5_energy": float(np.mean(values["final_whitened_pair_top5_energy"])) if values["final_whitened_pair_top5_energy"] else None,
                "mean_final_whitened_pair_effective_rank": float(np.mean(values["final_whitened_pair_effective_rank"])) if values["final_whitened_pair_effective_rank"] else None,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed-rank low-rank output-head classification experiment on frozen CIFAR-100 features."
    )
    parser.add_argument("--data-root", default="data/cifar100")
    parser.add_argument("--feature-cache-dir", default="data/cifar100/features")
    parser.add_argument("--results-dir", default="results/cifar100_fixed_rank_head")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--ranks", type=int, nargs="+", default=None)
    parser.add_argument(
        "--gauge-alpha",
        type=float,
        default=1.0,
        help="Initialize equivalent fixed-rank representatives as (alpha B0, A0 / alpha).",
    )
    parser.add_argument("--init-mode", choices=["random", "class_mean_svd"], default="random")
    parser.add_argument("--methods", nargs="+", default=None, choices=list(METHODS.keys()))
    parser.add_argument("--lr-grid", type=float, nargs="+", default=None)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bias-weight-decay", type=float, default=1e-4)
    parser.add_argument("--bias-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--feature-l2-normalize", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.seeds is None:
        args.seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    if args.ranks is None:
        args.ranks = SMOKE_RANKS if args.smoke else DEFAULT_RANKS
    if args.methods is None:
        args.methods = DEFAULT_METHODS
    if args.lr_grid is None:
        args.lr_grid = SMOKE_LR_GRID if args.smoke else DEFAULT_LR_GRID
    if args.smoke:
        args.synthetic = True
        args.max_epochs = min(args.max_epochs, 5)
        args.batch_size = min(args.batch_size, 128)
        args.eval_batch_size = min(args.eval_batch_size, 256)
    args.seeds = [int(v) for v in args.seeds]
    args.ranks = [int(v) for v in args.ranks]
    return args


def main() -> None:
    args = normalize_args(parse_args())
    setup_seed(args.seed)
    device = torch.device(args.device)
    results_dir = ensure_dir(args.results_dir)
    data = load_feature_data(args, device)

    save_json(
        results_dir / "run_config.json",
        {
            "data_source": data.source,
            "feature_dim": data.feature_dim,
            "n_classes": data.n_classes,
            "n_train_examples": int(data.train_labels.numel()),
            "n_test_examples": int(data.test_labels.numel()),
            "methods": args.methods,
            "ranks": args.ranks,
            "gauge_alpha": args.gauge_alpha,
            "init_mode": args.init_mode,
            "seeds": args.seeds,
            "lr_grid": args.lr_grid,
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
            "smoke": args.smoke,
            "device": str(device),
            "hardware": hardware_info(),
        },
    )

    shared_dir = ensure_dir(results_dir / "shared")
    for seed in args.seeds:
        split = split_features(
            data,
            seed=seed,
            validation_fraction=args.validation_fraction,
            device=device,
        )
        for rank in args.ranks:
            init_state = init_head(
                data.feature_dim,
                data.n_classes,
                rank,
                seed,
                device,
                gauge_alpha=args.gauge_alpha,
                init_mode=args.init_mode,
                split=split,
            )
            save_shared_artifacts(shared_dir, rank, seed, split, init_state)
            for method in args.methods:
                for lr in args.lr_grid:
                    run_dir = ensure_dir(results_dir / f"seed{seed}" / f"rank{rank}" / method / f"lr_{format_lr(lr)}")
                    init_path = shared_dir / f"init_seed{seed}_rank{rank}.pt"
                    config = {
                        "seed": seed,
                        "rank": rank,
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
                    final_state, logs, summary = run_one_training(
                        split,
                        init_state,
                        method=method,
                        lr=float(lr),
                        max_epochs=args.max_epochs,
                        batch_size=args.batch_size,
                        weight_decay=args.weight_decay,
                        bias_weight_decay=args.bias_weight_decay,
                        bias_lr_multiplier=args.bias_lr_multiplier,
                        eval_batch_size=args.eval_batch_size,
                        batch_seed_base=500_000 + 10_000 * seed + 100 * rank,
                    )
                    save_run_outputs(run_dir, config, init_path, logs, final_state, summary)

    sweep_rows_all = []
    best_lr_payload: dict[str, dict[str, float]] = {}
    aggregate_rows = []
    for rank in args.ranks:
        sweep_rows, best = rank_sweep_summary(results_dir, rank, args.methods, args.seeds, args.lr_grid)
        sweep_rows_all.extend(sweep_rows)
        best_lr_payload[f"rank{rank}"] = best
        aggregate_rows.extend(aggregate_best_runs(results_dir, rank, best, args.seeds))

    write_csv(results_dir / "lr_sweep_summary.csv", sweep_rows_all)
    save_json(results_dir / "global_best_lr.json", best_lr_payload)
    write_csv(results_dir / "mean_test_accuracy_table.csv", aggregate_rows)


if __name__ == "__main__":
    main()
