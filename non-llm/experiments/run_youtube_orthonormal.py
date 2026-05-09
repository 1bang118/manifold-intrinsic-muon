from __future__ import annotations

import argparse
import csv
import itertools
import json
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

from src.manifolds.grassmann import (
    orthonormalize,
    retraction_qr,
)
from src.metrics import compute_metrics_grassmann
from src.objectives.grassmann_barycenter import (
    LOSS_MODES as GRASSMANN_LOSS_MODES,
    barycenter_egrad,
    barycenter_loss,
    extrinsic_projector_mean,
    grassmann_distance,
)
from src.utils import (
    DEFAULT_DTYPE,
    ensure_dir,
    format_lr,
    hardware_info,
    matrix_power_symmetric,
    ortho,
    save_json,
    setup_seed,
    sym,
    write_csv,
)


torch.set_default_dtype(DEFAULT_DTYPE)


StepFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
RetrFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]

ITERATIVE_METHODS: dict[str, dict[str, Any]] = {
    "rgd": {"step_fn": None, "norm": "frobenius"},
    "imuon": {"step_fn": None, "norm": "spectral"},
    "imuon_nu": {"step_fn": None, "norm": "nuclear"},
    "egd": {"step_fn": None, "norm": "frobenius"},
    "muon": {"step_fn": None, "norm": "spectral"},
    "numuon": {"step_fn": None, "norm": "nuclear"},
}
BASELINE_METHODS = {"gda", "pml"}
ALL_METHODS = list(ITERATIVE_METHODS.keys()) + ["gda", "pml"]
DEFAULT_LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
SMOKE_LR_GRID = [1e-2]


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    label: str
    feature_path: str
    subspace: torch.Tensor


def _horizontal_project_orthonormal(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return z - y @ (y.transpose(-1, -2) @ z)


def _safe_normalize(x: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    norm = torch.linalg.norm(x, ord="fro")
    if not torch.isfinite(norm) or float(norm.item()) <= eps:
        return torch.zeros_like(x)
    return x / norm


def rgd_step_orthonormal(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = _horizontal_project_orthonormal(y, egrad)
    return _safe_normalize(h)


def imuon_step_orthonormal(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = _horizontal_project_orthonormal(y, egrad)
    if float(torch.linalg.norm(h, ord="fro").item()) <= 1e-14:
        return torch.zeros_like(y)
    return ortho(h)


def imuon_nu_step_orthonormal(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = _horizontal_project_orthonormal(y, egrad)
    u, s, vh = torch.linalg.svd(h, full_matrices=False)
    if s.numel() == 0 or float(s[0].item()) <= 1e-14:
        return torch.zeros_like(y)
    return u[:, :1] @ vh[:1, :]


def egd_step(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = _horizontal_project_orthonormal(y, egrad)
    return _safe_normalize(h)


def muon_step(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = _horizontal_project_orthonormal(y, egrad)
    if float(torch.linalg.norm(h, ord="fro").item()) <= 1e-14:
        return torch.zeros_like(y)
    return ortho(h)


def numuon_step(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = _horizontal_project_orthonormal(y, egrad)
    u, s, vh = torch.linalg.svd(h, full_matrices=False)
    if s.numel() == 0 or float(s[0].item()) <= 1e-14:
        return torch.zeros_like(y)
    return u[:, :1] @ vh[:1, :]


ITERATIVE_METHODS["rgd"]["step_fn"] = rgd_step_orthonormal
ITERATIVE_METHODS["imuon"]["step_fn"] = imuon_step_orthonormal
ITERATIVE_METHODS["imuon_nu"]["step_fn"] = imuon_nu_step_orthonormal
ITERATIVE_METHODS["egd"]["step_fn"] = egd_step
ITERATIVE_METHODS["muon"]["step_fn"] = muon_step
ITERATIVE_METHODS["numuon"]["step_fn"] = numuon_step


def label_key(label: Any) -> str:
    return str(label)


def safe_label(label: Any) -> str:
    safe = label_key(label)
    for old, new in [("/", "_"), (" ", "_"), (":", "_"), ("\\", "_")]:
        safe = safe.replace(old, new)
    return safe


def load_feature_tensor(path: Path, device: torch.device) -> torch.Tensor:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        for key in ("features", "feats", "x"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, torch.Tensor):
        payload = torch.as_tensor(payload)
    if payload.ndim != 2:
        raise ValueError(f"{path} should contain a 2D (n_frames, feature_dim) tensor, got {tuple(payload.shape)}")
    return payload.to(device=device, dtype=DEFAULT_DTYPE)


def features_to_subspace(features: torch.Tensor, k: int) -> torch.Tensor | None:
    if features.shape[0] < k or features.shape[1] < k:
        return None
    centered = features - features.mean(dim=0, keepdim=True)
    if float(torch.linalg.norm(centered, ord="fro").item()) <= 1e-14:
        return None
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < k:
        return None
    return orthonormalize(vh[:k, :].transpose(-1, -2))


def iter_feature_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("features.pt") if path.is_file())


def resolve_feature_dir(value: str) -> Path:
    if value != "auto":
        return Path(value)
    candidates = [
        Path("data/ytfaces/features"),
        Path("data/features"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_subspace_records(
    data_dir: Path,
    k: int,
    device: torch.device,
    max_subjects: int | None = None,
    max_clips_per_subject: int | None = None,
) -> list[ClipRecord]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} does not exist. Run experiments/extract_features.py after downloading YouTube Faces."
        )

    feature_files = iter_feature_files(data_dir)
    if not feature_files:
        raise FileNotFoundError(f"no features.pt files found under {data_dir}")

    files_by_subject: dict[str, list[Path]] = {}
    for feature_path in feature_files:
        rel_parent = feature_path.parent.relative_to(data_dir)
        if not rel_parent.parts:
            continue
        files_by_subject.setdefault(rel_parent.parts[0], []).append(feature_path)

    subjects = sorted(files_by_subject, key=lambda label: (-len(files_by_subject[label]), label))
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    subject_counts: dict[str, int] = {}
    records: list[ClipRecord] = []
    for label in subjects:
        for feature_path in sorted(files_by_subject[label]):
            if max_clips_per_subject is not None and subject_counts.get(label, 0) >= max_clips_per_subject:
                break
            rel_parent = feature_path.parent.relative_to(data_dir)
            features = load_feature_tensor(feature_path, device=device)
            subspace = features_to_subspace(features, k)
            if subspace is None:
                continue
            subject_counts[label] = subject_counts.get(label, 0) + 1
            clip_id = str(rel_parent)
            records.append(ClipRecord(clip_id=clip_id, label=label, feature_path=str(feature_path), subspace=subspace))

    eligible = {label for label, count in subject_counts.items() if count >= 2}
    records = [record for record in records if record.label in eligible]
    if len(eligible) < 2:
        raise RuntimeError(f"need at least two subjects with two clips each; found {subject_counts}")
    return records


def synthetic_records(seed: int, k: int, device: torch.device) -> list[ClipRecord]:
    generator = torch.Generator(device=device).manual_seed(seed)
    m = 14
    n_classes = 3
    clips_per_class = 6
    records: list[ClipRecord] = []
    for cls in range(n_classes):
        base = orthonormalize(torch.randn(m, k, generator=generator, device=device, dtype=DEFAULT_DTYPE))
        for clip in range(clips_per_class):
            noisy = orthonormalize(base + 0.15 * torch.randn(m, k, generator=generator, device=device, dtype=DEFAULT_DTYPE))
            records.append(
                ClipRecord(
                    clip_id=f"class_{cls:02d}/clip_{clip:02d}",
                    label=f"class_{cls:02d}",
                    feature_path="synthetic",
                    subspace=noisy,
                )
            )
    return records


def make_split(records: list[ClipRecord], seed: int, train_frac: float) -> tuple[list[int], list[int], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        by_label.setdefault(record.label, []).append(idx)

    train_idx: list[int] = []
    test_idx: list[int] = []
    split_by_label: dict[str, dict[str, list[str]]] = {}
    for label in sorted(by_label):
        indices = np.asarray(by_label[label], dtype=int)
        if len(indices) < 2:
            continue
        rng.shuffle(indices)
        n_train = int(math.floor(train_frac * len(indices)))
        n_train = min(max(1, n_train), len(indices) - 1)
        label_train = indices[:n_train].tolist()
        label_test = indices[n_train:].tolist()
        train_idx.extend(label_train)
        test_idx.extend(label_test)
        split_by_label[label] = {
            "train": [records[i].clip_id for i in label_train],
            "test": [records[i].clip_id for i in label_test],
        }

    train_idx.sort()
    test_idx.sort()
    return train_idx, test_idx, {"train_frac": train_frac, "by_label": split_by_label}


def save_split(path: Path, seed: int, k: int, split_summary: dict[str, Any]) -> None:
    save_json(path, {"seed": seed, "k": k, **split_summary})


def selected_records(records: list[ClipRecord], indices: list[int]) -> list[ClipRecord]:
    return [records[i] for i in indices]


def records_to_tensor(records: list[ClipRecord]) -> torch.Tensor:
    return torch.stack([record.subspace for record in records])


def initializer_gauge_matrix(k: int, dtype: torch.dtype, device: torch.device, column_condition: float) -> torch.Tensor:
    if column_condition <= 1.0:
        return torch.eye(k, dtype=dtype, device=device)
    exponents = torch.linspace(-0.5, 0.5, k, dtype=dtype, device=device)
    return torch.diag(torch.as_tensor(float(column_condition), dtype=dtype, device=device).pow(exponents))


def save_initializers(
    split_dir: Path,
    train_records: list[ClipRecord],
    classes: list[str],
    column_condition: float,
) -> dict[str, torch.Tensor]:
    init_dir = ensure_dir(split_dir / "initializers")
    initializers: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for cls in classes:
        class_records = [record for record in train_records if record.label == cls]
        raw_init = class_records[0].subspace.detach().clone()
        gauge = initializer_gauge_matrix(raw_init.shape[1], raw_init.dtype, raw_init.device, column_condition)
        init = raw_init @ gauge
        initializers[cls] = init
        rows.append(
            {
                "class_label": cls,
                "init_clip_id": class_records[0].clip_id,
                "n_train": len(class_records),
                "initializer_column_condition": column_condition,
                "gram_condition_number": float(torch.linalg.cond(init.transpose(-1, -2) @ init).item()),
            }
        )
    torch.save({k: v.detach().cpu() for k, v in initializers.items()}, init_dir / "init_checkpoint.pt")
    write_csv(init_dir / "init_summary.csv", rows)
    return initializers


def convergence_row(
    method: str,
    lr: float | None,
    class_label: str | None,
    iteration: int,
    objective: float,
    metrics: dict[str, float],
    wall_time_sec: float,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "lr": lr,
        "class_label": class_label,
        "iter": iteration,
        "objective": objective,
        "wall_time_sec": wall_time_sec,
        "status": status,
    }
    row.update(metrics)
    if error:
        row["error"] = error
    return row


def compute_karcher_mean(
    subspaces: torch.Tensor,
    y0: torch.Tensor,
    method: str,
    class_label: str,
    step_fn: StepFn,
    norm_type: str,
    lr: float,
    max_iters: int,
    tol: float,
    retraction: RetrFn,
    loss_mode: str,
    loss_beta: float,
    loss_eps: float,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    y = y0.clone()
    logs: list[dict[str, Any]] = []
    status = "max_iters"
    error: str | None = None
    converged = False
    iters_done = 0

    for iteration in range(max_iters):
        t0 = time.perf_counter()
        try:
            objective = float(
                barycenter_loss(
                    y,
                    subspaces,
                    loss_mode=loss_mode,
                    loss_beta=loss_beta,
                    loss_eps=loss_eps,
                ).item()
            )
            egrad = barycenter_egrad(
                y,
                subspaces,
                loss_mode=loss_mode,
                loss_beta=loss_beta,
                loss_eps=loss_eps,
            )
            xi = step_fn(y, egrad)
            metrics = compute_metrics_grassmann(y, egrad, xi, norm_type=norm_type)
            wall = time.perf_counter() - t0
            logs.append(convergence_row(method, lr, class_label, iteration, objective, metrics, wall))
            iters_done = iteration + 1
            if not math.isfinite(objective) or any(
                not math.isfinite(float(v)) for v in metrics.values() if isinstance(v, (int, float))
            ):
                status = "failed"
                error = "non-finite objective or metric"
                break
            if metrics["rgrad_norm"] < tol:
                status = "converged"
                converged = True
                break
            y = retraction(y, xi, lr).detach()
        except Exception as exc:
            status = "failed"
            error = repr(exc)
            logs.append(
                convergence_row(
                    method,
                    lr,
                    class_label,
                    iteration,
                    float("nan"),
                    {"rgrad_norm": float("nan"), "dual_norm_H": float("nan"), "Z_norm_sq": float("nan")},
                    time.perf_counter() - t0,
                    status=status,
                    error=error,
                )
            )
            iters_done = iteration + 1
            break

    summary = {
        "status": status,
        "error": error,
        "converged": converged,
        "n_iters": iters_done,
        "final_objective": None if not logs else logs[-1]["objective"],
        "final_rgrad_norm": None if not logs else logs[-1].get("rgrad_norm"),
    }
    return y.detach(), logs, summary


def classify_nearest_mean(
    class_means: dict[str, torch.Tensor],
    test_records: list[ClipRecord],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    classes = sorted(class_means.keys(), key=label_key)
    rows: list[dict[str, Any]] = []
    correct = 0
    for record in test_records:
        dists = [float(grassmann_distance(record.subspace, class_means[cls]).item()) for cls in classes]
        pred = classes[int(np.argmin(dists))]
        correct += int(pred == record.label)
        row = {
            "clip_id": record.clip_id,
            "true_label": record.label,
            "predicted_label": pred,
            "min_distance": min(dists),
        }
        for cls, dist in zip(classes, dists):
            row[f"distance_to_{safe_label(cls)}"] = dist
        rows.append(row)
    accuracy = correct / max(1, len(test_records))
    return rows, {"accuracy": float(accuracy)}


def save_class_means(out_dir: Path, class_means: dict[str, torch.Tensor]) -> None:
    class_dir = ensure_dir(out_dir / "class_means")
    for cls, mean in class_means.items():
        torch.save(mean.detach().cpu(), class_dir / f"class_{safe_label(cls)}.pt")


def save_run_outputs(
    out_dir: Path,
    config: dict[str, Any],
    init_path: Path,
    class_means: dict[str, torch.Tensor],
    convergence_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    lr_sweep_rows: list[dict[str, Any]],
    extra_tensors: dict[str, torch.Tensor] | None = None,
) -> None:
    ensure_dir(out_dir)
    save_json(out_dir / "config.json", config)
    torch.save({"shared_init_path": str(init_path)}, out_dir / "init_checkpoint.pt")
    save_class_means(out_dir, class_means)
    torch.save({k: v.detach().cpu() for k, v in class_means.items()}, out_dir / "class_means.pt")
    if extra_tensors:
        for name, tensor in extra_tensors.items():
            torch.save(tensor.detach().cpu(), out_dir / name)
    write_csv(out_dir / "convergence.csv", convergence_rows)
    write_csv(out_dir / "predictions.csv", prediction_rows)
    save_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "lr_sweep.csv", lr_sweep_rows)


def run_tier1_method(
    method: str,
    method_dir: Path,
    seed: int,
    k: int,
    train_records: list[ClipRecord],
    test_records: list[ClipRecord],
    classes: list[str],
    initializers: dict[str, torch.Tensor],
    init_path: Path,
    lrs: list[float],
    max_iters: int,
    tol: float,
    retraction: RetrFn,
    base_config: dict[str, Any],
    loss_mode: str,
    loss_beta: float,
    loss_eps: float,
) -> dict[str, Any]:
    method_cfg = ITERATIVE_METHODS[method]
    step_fn: StepFn = method_cfg["step_fn"]
    norm_type = str(method_cfg["norm"])
    all_convergence: list[dict[str, Any]] = []
    lr_sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    t_method = time.perf_counter()

    for lr in lrs:
        lr_dir = ensure_dir(method_dir / "lr_runs" / f"lr_{format_lr(lr)}")
        class_means: dict[str, torch.Tensor] = {}
        convergence_rows: list[dict[str, Any]] = []
        per_class: dict[str, Any] = {}
        status = "ok"
        error = None
        t_lr = time.perf_counter()

        for cls in classes:
            class_records = [record for record in train_records if record.label == cls]
            subspaces = records_to_tensor(class_records)
            mean, rows, class_summary = compute_karcher_mean(
                subspaces,
                initializers[cls],
                method=method,
                class_label=cls,
                step_fn=step_fn,
                norm_type=norm_type,
                lr=lr,
                max_iters=max_iters,
                tol=tol,
                retraction=retraction,
                loss_mode=loss_mode,
                loss_beta=loss_beta,
                loss_eps=loss_eps,
            )
            class_means[cls] = mean
            convergence_rows.extend(rows)
            per_class[cls] = class_summary
            if class_summary["status"] == "failed":
                status = "failed"
                error = class_summary["error"]
                break

        all_convergence.extend(convergence_rows)
        if status == "ok":
            predictions, metrics = classify_nearest_mean(class_means, test_records)
            accuracy = metrics["accuracy"]
        else:
            predictions = []
            accuracy = float("nan")

        n_iters_to_converge = max((v["n_iters"] for v in per_class.values()), default=0)
        lr_summary = {
            "method": method,
            "seed": seed,
            "k": k,
            "status": status,
            "error": error,
            "accuracy": accuracy,
            "best_lr": lr,
            "n_iters_to_converge": n_iters_to_converge,
            "runtime_sec": time.perf_counter() - t_lr,
        }
        save_run_outputs(
            lr_dir,
            dict(base_config, method=method, method_type="tier1_karcher", lr=lr, norm_type=norm_type),
            init_path,
            class_means,
            convergence_rows,
            predictions,
            lr_summary,
            [{"method": method, "seed": seed, "k": k, "lr": lr, "status": status, "accuracy": accuracy}],
        )
        lr_sweep.append(
            {
                "method": method,
                "seed": seed,
                "k": k,
                "lr": lr,
                "status": status,
                "accuracy": accuracy,
                "n_iters_to_converge": n_iters_to_converge,
                "runtime_sec": lr_summary["runtime_sec"],
                "error": error,
            }
        )
        if status == "ok" and math.isfinite(accuracy):
            if best is None or accuracy > best["summary"]["accuracy"]:
                best = {"lr": lr, "class_means": class_means, "predictions": predictions, "summary": lr_summary}

    if best is None:
        top_summary = {
            "method": method,
            "seed": seed,
            "k": k,
            "status": "failed",
            "accuracy": float("nan"),
            "best_lr": None,
            "n_iters_to_converge": None,
            "runtime_sec": time.perf_counter() - t_method,
        }
        best_class_means = {}
        best_predictions: list[dict[str, Any]] = []
    else:
        top_summary = dict(best["summary"])
        top_summary.update({"best_lr": best["lr"], "runtime_sec": time.perf_counter() - t_method})
        best_class_means = best["class_means"]
        best_predictions = best["predictions"]

    save_run_outputs(
        method_dir,
        dict(base_config, method=method, method_type="tier1_karcher", lrs=lrs, max_iters=max_iters, tol=tol),
        init_path,
        best_class_means,
        all_convergence,
        best_predictions,
        top_summary,
        lr_sweep,
    )
    return top_summary


def projection_scatter_gda(train_records: list[ClipRecord], classes: list[str], out_dim: int) -> torch.Tensor:
    first = train_records[0].subspace
    m = first.shape[0]
    eye = torch.eye(m, dtype=first.dtype, device=first.device)

    class_sums: dict[str, torch.Tensor] = {cls: torch.zeros(m, m, dtype=first.dtype, device=first.device) for cls in classes}
    class_counts: dict[str, int] = {cls: 0 for cls in classes}
    grand_sum = torch.zeros(m, m, dtype=first.dtype, device=first.device)
    for record in train_records:
        q = orthonormalize(record.subspace)
        projector = q @ q.transpose(-1, -2)
        class_sums[record.label] = class_sums[record.label] + projector
        class_counts[record.label] += 1
        grand_sum = grand_sum + projector

    grand = grand_sum / max(1, len(train_records))
    class_means = {
        cls: class_sums[cls] / max(1, class_counts[cls])
        for cls in classes
    }
    sw = torch.zeros_like(grand)
    sb = torch.zeros_like(grand)
    for record in train_records:
        q = orthonormalize(record.subspace)
        projector = q @ q.transpose(-1, -2)
        diff = projector - class_means[record.label]
        sw = sw + diff @ diff
    for cls in classes:
        diff_mean = class_means[cls] - grand
        sb = sb + class_counts[cls] * (diff_mean @ diff_mean)

    sw = sym(sw / max(1, len(train_records))) + 1e-6 * eye
    sb = sym(sb / max(1, len(train_records)))
    sw_inv_half = matrix_power_symmetric(sw, -0.5)
    whitened = sym(sw_inv_half @ sb @ sw_inv_half)
    _, eigvecs = torch.linalg.eigh(whitened)
    w = sw_inv_half @ eigvecs[:, -out_dim:]
    return orthonormalize(w)


def project_record_subspace(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return orthonormalize(w.transpose(-1, -2) @ y)


def projected_records(records: list[ClipRecord], w: torch.Tensor) -> list[ClipRecord]:
    return [
        ClipRecord(
            clip_id=record.clip_id,
            label=record.label,
            feature_path=record.feature_path,
            subspace=project_record_subspace(w, record.subspace),
        )
        for record in records
    ]


def extrinsic_class_means(records: list[ClipRecord], classes: list[str], k: int) -> dict[str, torch.Tensor]:
    means: dict[str, torch.Tensor] = {}
    for cls in classes:
        cls_records = [record for record in records if record.label == cls]
        means[cls] = extrinsic_projector_mean(records_to_tensor(cls_records), k=k)
    return means


def baseline_convergence_rows(
    method: str,
    class_means: dict[str, torch.Tensor],
    train_records: list[ClipRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cls, mean in class_means.items():
        class_records = [record for record in train_records if record.label == cls]
        t0 = time.perf_counter()
        objective = float(barycenter_loss(mean, records_to_tensor(class_records)).item())
        rows.append(
            convergence_row(
                method,
                None,
                cls,
                0,
                objective,
                {"rgrad_norm": float("nan"), "dual_norm_H": float("nan"), "Z_norm_sq": float("nan")},
                time.perf_counter() - t0,
                status="closed_form",
            )
        )
    return rows


def run_gda_method(
    method_dir: Path,
    seed: int,
    k: int,
    train_records: list[ClipRecord],
    test_records: list[ClipRecord],
    classes: list[str],
    init_path: Path,
    projection_dim: int,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    m = train_records[0].subspace.shape[0]
    out_dim = min(max(projection_dim, k), m)
    w = projection_scatter_gda(train_records, classes, out_dim=out_dim)
    train_projected = projected_records(train_records, w)
    test_projected = projected_records(test_records, w)
    class_means = extrinsic_class_means(train_projected, classes, k=k)
    predictions, metrics = classify_nearest_mean(class_means, test_projected)
    convergence = baseline_convergence_rows("gda", class_means, train_projected)
    summary = {
        "method": "gda",
        "seed": seed,
        "k": k,
        "status": "ok",
        "accuracy": metrics["accuracy"],
        "best_lr": None,
        "n_iters_to_converge": 0,
        "projection_dim": out_dim,
        "runtime_sec": time.perf_counter() - t0,
    }
    save_run_outputs(
        method_dir,
        dict(base_config, method="gda", method_type="projection_scatter_gda", projection_dim=out_dim),
        init_path,
        class_means,
        convergence,
        predictions,
        summary,
        [{"method": "gda", "seed": seed, "k": k, "lr": None, "status": "ok", "accuracy": metrics["accuracy"]}],
        extra_tensors={"projection.pt": w},
    )
    return summary


def projected_grassmann_distance_sq(w: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    q1 = orthonormalize(w.transpose(-1, -2) @ y1)
    q2 = orthonormalize(w.transpose(-1, -2) @ y2)
    svdvals = torch.linalg.svdvals(q1.transpose(-1, -2) @ q2).clamp(0.0, 1.0)
    svdvals = torch.where(svdvals > 1.0 - 1e-12, torch.ones_like(svdvals), svdvals)
    theta = torch.acos(svdvals)
    return torch.sum(theta * theta)


def make_pml_pairs(
    train_records: list[ClipRecord],
    classes: list[str],
    seed: int,
    max_pos_pairs_per_class: int,
    max_negatives_per_class: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    class_to_indices = {cls: [i for i, record in enumerate(train_records) if record.label == cls] for cls in classes}
    positive: list[tuple[int, int]] = []
    negative: list[tuple[int, int]] = []
    all_indices = np.arange(len(train_records), dtype=int)
    for cls in classes:
        indices = class_to_indices[cls]
        pairs = list(itertools.combinations(indices, 2))
        if len(pairs) > max_pos_pairs_per_class:
            chosen = rng.choice(len(pairs), size=max_pos_pairs_per_class, replace=False)
            pairs = [pairs[int(i)] for i in chosen]
        positive.extend(pairs)

        other = np.asarray([i for i in all_indices if train_records[int(i)].label != cls], dtype=int)
        anchors = np.asarray(indices, dtype=int)
        if len(other) == 0 or len(anchors) == 0:
            continue
        n_neg = min(max_negatives_per_class, len(anchors) * len(other))
        seen: set[tuple[int, int]] = set()
        while len(seen) < n_neg:
            i = int(rng.choice(anchors))
            j = int(rng.choice(other))
            seen.add((i, j))
        negative.extend(sorted(seen))
    return positive, negative


def pml_objective(
    w: torch.Tensor,
    train_records: list[ClipRecord],
    positive_pairs: list[tuple[int, int]],
    negative_pairs: list[tuple[int, int]],
) -> torch.Tensor:
    total_pos = torch.zeros((), dtype=w.dtype, device=w.device)
    for i, j in positive_pairs:
        total_pos = total_pos + projected_grassmann_distance_sq(w, train_records[i].subspace, train_records[j].subspace)
    total_neg = torch.zeros((), dtype=w.dtype, device=w.device)
    for i, j in negative_pairs:
        total_neg = total_neg + projected_grassmann_distance_sq(w, train_records[i].subspace, train_records[j].subspace)
    pos = total_pos / max(1, len(positive_pairs))
    neg = total_neg / max(1, len(negative_pairs))
    return pos - neg


def run_pml_single_lr(
    w0: torch.Tensor,
    lr: float,
    seed: int,
    k: int,
    train_records: list[ClipRecord],
    test_records: list[ClipRecord],
    classes: list[str],
    positive_pairs: list[tuple[int, int]],
    negative_pairs: list[tuple[int, int]],
    max_iters: int,
    base_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    w = w0.detach().clone()
    logs: list[dict[str, Any]] = []
    status = "max_iters"
    error = None

    for iteration in range(max_iters):
        t0 = time.perf_counter()
        try:
            w_var = w.detach().clone().requires_grad_(True)
            objective = pml_objective(w_var, train_records, positive_pairs, negative_pairs)
            objective.backward()
            egrad = w_var.grad.detach()
            xi = rgd_step_orthonormal(w, egrad)
            metrics = compute_metrics_grassmann(w, egrad, xi, norm_type="frobenius")
            logs.append(convergence_row("pml", lr, None, iteration, float(objective.detach().item()), metrics, time.perf_counter() - t0))
            w = retraction_qr(w, xi, eta=lr).detach()
            status = "ok"
        except Exception as exc:
            status = "failed"
            error = repr(exc)
            logs.append(
                convergence_row(
                    "pml",
                    lr,
                    None,
                    iteration,
                    float("nan"),
                    {"rgrad_norm": float("nan"), "dual_norm_H": float("nan"), "Z_norm_sq": float("nan")},
                    time.perf_counter() - t0,
                    status=status,
                    error=error,
                )
            )
            break

    train_projected = projected_records(train_records, w)
    test_projected = projected_records(test_records, w)
    class_means = extrinsic_class_means(train_projected, classes, k=k)
    predictions, metrics = classify_nearest_mean(class_means, test_projected)
    summary = {
        "method": "pml",
        "seed": seed,
        "k": k,
        "status": status,
        "error": error,
        "accuracy": metrics["accuracy"] if status != "failed" else float("nan"),
        "best_lr": lr,
        "n_iters_to_converge": len(logs),
        "projection_dim": int(w.shape[1]),
        "config": {
            "positive_pairs": len(positive_pairs),
            "negative_pairs": len(negative_pairs),
            "pml_max_iters": max_iters,
        },
        "base_config": base_config,
    }
    return w, class_means, logs, predictions, summary


def run_pml_method(
    method_dir: Path,
    seed: int,
    k: int,
    train_records: list[ClipRecord],
    test_records: list[ClipRecord],
    classes: list[str],
    init_path: Path,
    lrs: list[float],
    projection_dim: int,
    pml_iters: int,
    max_pos_pairs_per_class: int,
    max_negatives_per_class: int,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    t_method = time.perf_counter()
    m = train_records[0].subspace.shape[0]
    out_dim = min(max(projection_dim, k), m)
    generator = torch.Generator(device=train_records[0].subspace.device).manual_seed(seed + 10_000 + k)
    w0 = orthonormalize(torch.randn(m, out_dim, generator=generator, device=train_records[0].subspace.device, dtype=DEFAULT_DTYPE))
    torch.save(w0.detach().cpu(), method_dir / "pml_projection_init.pt")
    positive_pairs, negative_pairs = make_pml_pairs(
        train_records,
        classes,
        seed=seed,
        max_pos_pairs_per_class=max_pos_pairs_per_class,
        max_negatives_per_class=max_negatives_per_class,
    )

    all_convergence: list[dict[str, Any]] = []
    lr_sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for lr in lrs:
        lr_dir = ensure_dir(method_dir / "lr_runs" / f"lr_{format_lr(lr)}")
        t_lr = time.perf_counter()
        w, class_means, logs, predictions, summary = run_pml_single_lr(
            w0,
            lr,
            seed,
            k,
            train_records,
            test_records,
            classes,
            positive_pairs,
            negative_pairs,
            pml_iters,
            base_config,
        )
        summary["runtime_sec"] = time.perf_counter() - t_lr
        all_convergence.extend(logs)
        save_run_outputs(
            lr_dir,
            dict(base_config, method="pml", method_type="projection_metric_learning", lr=lr, projection_dim=out_dim),
            init_path,
            class_means,
            logs,
            predictions,
            summary,
            [{"method": "pml", "seed": seed, "k": k, "lr": lr, "status": summary["status"], "accuracy": summary["accuracy"]}],
            extra_tensors={"projection.pt": w},
        )
        lr_sweep.append(
            {
                "method": "pml",
                "seed": seed,
                "k": k,
                "lr": lr,
                "status": summary["status"],
                "accuracy": summary["accuracy"],
                "n_iters_to_converge": summary["n_iters_to_converge"],
                "runtime_sec": summary["runtime_sec"],
                "error": summary["error"],
            }
        )
        if summary["status"] != "failed" and math.isfinite(float(summary["accuracy"])):
            if best is None or summary["accuracy"] > best["summary"]["accuracy"]:
                best = {"lr": lr, "w": w, "class_means": class_means, "logs": logs, "predictions": predictions, "summary": summary}

    if best is None:
        top_summary = {
            "method": "pml",
            "seed": seed,
            "k": k,
            "status": "failed",
            "accuracy": float("nan"),
            "best_lr": None,
            "n_iters_to_converge": None,
            "runtime_sec": time.perf_counter() - t_method,
        }
        best_class_means = {}
        best_predictions: list[dict[str, Any]] = []
        best_projection = w0
    else:
        top_summary = dict(best["summary"])
        top_summary.update({"best_lr": best["lr"], "runtime_sec": time.perf_counter() - t_method})
        best_class_means = best["class_means"]
        best_predictions = best["predictions"]
        best_projection = best["w"]

    save_run_outputs(
        method_dir,
        dict(
            base_config,
            method="pml",
            method_type="projection_metric_learning",
            lrs=lrs,
            projection_dim=out_dim,
            pml_iters=pml_iters,
        ),
        init_path,
        best_class_means,
        all_convergence,
        best_predictions,
        top_summary,
        lr_sweep,
        extra_tensors={"projection.pt": best_projection},
    )
    return top_summary


def run_seed_k(seed: int, k: int, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    split_dir = ensure_dir(Path(args.results_dir) / f"seed{seed}" / f"k{k}")
    if args.synthetic_smoke:
        records = synthetic_records(seed=seed, k=k, device=device)
        data_source = "synthetic_smoke"
    else:
        feature_dir = resolve_feature_dir(args.data_dir)
        records = load_subspace_records(
            feature_dir,
            k=k,
            device=device,
            max_subjects=args.max_subjects,
            max_clips_per_subject=args.max_clips_per_subject,
        )
        data_source = str(feature_dir)

    train_idx, test_idx, split_summary = make_split(records, seed=seed, train_frac=args.train_frac)
    save_split(split_dir / f"split_seed{seed}.json", seed, k, split_summary)
    train_records = selected_records(records, train_idx)
    test_records = selected_records(records, test_idx)
    classes = sorted({record.label for record in train_records}, key=label_key)
    initializers = save_initializers(split_dir, train_records, classes, args.init_gauge_condition)
    init_path = split_dir / "initializers" / "init_checkpoint.pt"
    retraction = retraction_qr
    retraction_name = "qr_stiefel_gauge"
    representation = "orthonormal_stiefel_gauge"

    data_summary = {
        "source": data_source,
        "n_records": len(records),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "n_classes": len(classes),
        "feature_dim": int(records[0].subspace.shape[0]),
        "subspace_dim": k,
    }
    save_json(split_dir / "data_summary.json", data_summary)
    base_config = {
        "seed": seed,
        "k": k,
        "dataset": "YouTube Faces features" if not args.synthetic_smoke else "synthetic_smoke",
        "feature_model": "ResNet-18 pool5",
        "train_frac": args.train_frac,
        "retraction": retraction_name,
        "grassmann_representation": representation,
        "init_gauge_column_condition": args.init_gauge_condition,
        "device": str(device),
        "hardware": hardware_info(),
        "data_summary": data_summary,
        "initializer_path": str(init_path),
        "split_path": str(split_dir / f"split_seed{seed}.json"),
        "loss_mode": args.loss_mode,
        "loss_beta": args.loss_beta,
        "loss_eps": args.loss_eps,
    }

    summaries: list[dict[str, Any]] = []
    for method in args.methods:
        method_dir = ensure_dir(split_dir / method)
        print(f"[seed {seed} k {k}] running {method}")
        if method in ITERATIVE_METHODS:
            summary = run_tier1_method(
                method,
                method_dir,
                seed,
                k,
                train_records,
                test_records,
                classes,
                initializers,
                init_path,
                args.lrs,
                args.max_iters,
                args.tol,
                retraction,
                base_config,
                args.loss_mode,
                args.loss_beta,
                args.loss_eps,
            )
        elif method == "gda":
            summary = run_gda_method(
                method_dir,
                seed,
                k,
                train_records,
                test_records,
                classes,
                init_path,
                args.projection_dim,
                base_config,
            )
        elif method == "pml":
            summary = run_pml_method(
                method_dir,
                seed,
                k,
                train_records,
                test_records,
                classes,
                init_path,
                args.lrs,
                args.projection_dim,
                args.pml_iters,
                args.pml_pos_pairs_per_class,
                args.pml_negatives_per_class,
                base_config,
            )
        else:
            raise ValueError(method)
        summaries.append(summary)
        print(f"[seed {seed} k {k}] {method}: status={summary.get('status')} acc={summary.get('accuracy')} best_lr={summary.get('best_lr')}")

    write_csv(split_dir / "summary.csv", summaries)
    return summaries


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def write_global_reports(results_dir: Path, seeds: list[int], ks: list[int], methods: list[str]) -> None:
    selection_rows: list[dict[str, Any]] = []
    best_by_method_k: dict[str, dict[str, Any]] = {}

    for k in ks:
        for method in methods:
            if method not in ITERATIVE_METHODS and method != "pml":
                continue
            grouped: dict[float, list[float]] = {}
            for seed in seeds:
                rows = _read_csv_rows(results_dir / f"seed{seed}" / f"k{k}" / method / "lr_sweep.csv")
                for row in rows:
                    lr = _finite_float(row.get("lr"))
                    acc = _finite_float(row.get("accuracy"))
                    if lr is None or acc is None or row.get("status") != "ok":
                        continue
                    grouped.setdefault(lr, []).append(acc)
            candidates: list[dict[str, Any]] = []
            for lr, values in sorted(grouped.items()):
                candidate = {
                    "method": method,
                    "k": k,
                    "lr": lr,
                    "n_splits": len(values),
                    "complete_seed_grid": len(values) == len(seeds),
                    "mean_accuracy": float(np.mean(values)) if values else float("nan"),
                    "std_accuracy": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
                candidates.append(candidate)
            if not candidates:
                continue
            complete = [row for row in candidates if row["complete_seed_grid"]]
            selectable = complete if complete else candidates
            best = max(selectable, key=lambda row: (row["mean_accuracy"], row["n_splits"], -row["lr"]))
            best_by_method_k[f"{method}/k{k}"] = best
            for row in candidates:
                tagged = dict(row)
                tagged["selected_global_lr"] = row["lr"] == best["lr"]
                selection_rows.append(tagged)

    write_csv(results_dir / "global_lr_selection.csv", selection_rows)
    save_json(results_dir / "global_best_lr.json", best_by_method_k)

    selected_rows: list[dict[str, Any]] = []
    for k in ks:
        for seed in seeds:
            for method in methods:
                method_dir = results_dir / f"seed{seed}" / f"k{k}" / method
                if method in ITERATIVE_METHODS or method == "pml":
                    best = best_by_method_k.get(f"{method}/k{k}")
                    selected_lr = None if best is None else float(best["lr"])
                    match = None
                    for row in _read_csv_rows(method_dir / "lr_sweep.csv"):
                        lr = _finite_float(row.get("lr"))
                        if selected_lr is not None and lr is not None and math.isclose(lr, selected_lr, rel_tol=0.0, abs_tol=1e-15):
                            match = row
                            break
                    selected_rows.append(
                        {
                            "seed": seed,
                            "k": k,
                            "method": method,
                            "selection_scope": "global",
                            "best_lr": selected_lr,
                            "status": None if match is None else match.get("status"),
                            "accuracy": float("nan") if match is None else _finite_float(match.get("accuracy")),
                        }
                    )
                else:
                    summary_path = method_dir / "summary.json"
                    if not summary_path.exists():
                        continue
                    with summary_path.open("r") as handle:
                        summary = json.load(handle)
                    selected_rows.append(
                        {
                            "seed": seed,
                            "k": k,
                            "method": method,
                            "selection_scope": "closed_form",
                            "best_lr": None,
                            "status": summary.get("status"),
                            "accuracy": summary.get("accuracy"),
                        }
                    )
    write_csv(results_dir / "global_selected_summary.csv", selected_rows)

    table_rows: list[dict[str, Any]] = []
    for k in ks:
        for method in methods:
            rows = [row for row in selected_rows if row["k"] == k and row["method"] == method and row["status"] == "ok"]
            accuracies = [float(row["accuracy"]) for row in rows if row["accuracy"] is not None and math.isfinite(float(row["accuracy"]))]
            if not accuracies:
                continue
            table_rows.append(
                {
                    "method": method,
                    "k": k,
                    "selected_global_lr": best_by_method_k.get(f"{method}/k{k}", {}).get("lr"),
                    "n_splits": len(accuracies),
                    "mean_accuracy": float(np.mean(accuracies)),
                    "std_accuracy": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
                }
            )
    write_csv(results_dir / "accuracy_table.csv", table_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the orthonormal-gauge YouTube Faces Grassmann experiments from the current experiment plan.")
    parser.add_argument(
        "--data-dir",
        default="auto",
        help="Feature root. 'auto' checks data/ytfaces/features then data/features.",
    )
    parser.add_argument("--results-dir", default="results/youtube_orthonormal", help="Output directory.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random 70/30 splits.")
    parser.add_argument("--seed-values", type=int, nargs="+", default=None, help="Explicit split seeds.")
    parser.add_argument("--ks", type=int, nargs="+", default=[3, 5, 8, 10], help="Subspace dimensions.")
    parser.add_argument("--methods", nargs="+", default=None, choices=ALL_METHODS, help="Methods to run.")
    parser.add_argument("--lrs", type=float, nargs="+", default=None, help="LR grid for iterative methods.")
    parser.add_argument("--max-iters", type=int, default=None, help="Max Karcher iterations per class.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Riemannian gradient norm convergence threshold.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Per-subject train split fraction.")
    parser.add_argument("--projection-dim", type=int, default=64, help="Projected dimension for GDA/PML baselines.")
    parser.add_argument("--pml-iters", type=int, default=None, help="PML iterations per LR.")
    parser.add_argument("--pml-pos-pairs-per-class", type=int, default=50, help="Positive PML pairs sampled per class.")
    parser.add_argument("--pml-negatives-per-class", type=int, default=50, help="Negative PML pairs sampled per class.")
    parser.add_argument(
        "--retraction",
        choices=["qr"],
        default="qr",
        help="Retraction for the orthonormal Grassmann task. Fixed to QR to match the current experiment plan.",
    )
    parser.add_argument(
        "--init-gauge-condition",
        type=float,
        default=None,
        help="Column condition number of the deterministic initializer representative. Defaults to 1 for the orthonormal QR gauge.",
    )
    parser.add_argument("--max-subjects", type=int, default=47, help="Use the subjects with the most clips; default matches the 47-subject plan.")
    parser.add_argument("--all-subjects", action="store_true", help="Use every eligible subject instead of the default 47.")
    parser.add_argument("--max-clips-per-subject", type=int, default=None, help="Optional clip cap per subject for debugging.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument(
        "--loss-mode",
        default="squared_geodesic",
        choices=sorted(GRASSMANN_LOSS_MODES),
        help="Unitary-invariant aggregation on principal angles for the Grassmann barycenter objective.",
    )
    parser.add_argument(
        "--loss-beta",
        type=float,
        default=0.25,
        help="Mixing weight used when --loss-mode=fro_softnuclear.",
    )
    parser.add_argument(
        "--loss-eps",
        type=float,
        default=1e-6,
        help="Smoothing epsilon used by soft_nuclear-style Grassmann losses.",
    )
    parser.add_argument("--synthetic-smoke", action="store_true", help="Use synthetic subspaces instead of YouTube features.")
    parser.add_argument("--smoke", action="store_true", help="Small one-seed, one-k smoke run.")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.all_subjects:
        args.max_subjects = None
    if args.smoke:
        if args.seed_values is None:
            args.seed_values = [0]
        args.seeds = len(args.seed_values)
        if args.ks == [3, 5, 8, 10]:
            args.ks = [3]
        if args.lrs is None:
            args.lrs = SMOKE_LR_GRID
        if args.max_iters is None:
            args.max_iters = 3
        if args.pml_iters is None:
            args.pml_iters = 3
    else:
        if args.seed_values is None:
            args.seed_values = list(range(args.seeds))
        if args.lrs is None:
            args.lrs = DEFAULT_LR_GRID
        if args.max_iters is None:
            args.max_iters = 100
        if args.pml_iters is None:
            args.pml_iters = 100
    if args.methods is None:
        args.methods = ALL_METHODS
    if args.init_gauge_condition is None:
        args.init_gauge_condition = 1.0
    if args.init_gauge_condition < 1.0:
        raise ValueError("--init-gauge-condition must be >= 1")
    if not (0.0 <= args.loss_beta <= 1.0):
        raise ValueError("--loss-beta must be in [0, 1]")
    if args.loss_eps <= 0.0:
        raise ValueError("--loss-eps must be positive")
    return args


def main() -> None:
    args = normalize_args(parse_args())
    setup_seed(0)
    device = torch.device(args.device)
    results_dir = ensure_dir(args.results_dir)
    save_json(
        results_dir / "run_config.json",
        {
            "seeds": args.seed_values,
            "ks": args.ks,
            "methods": args.methods,
            "lrs": args.lrs,
            "max_iters": args.max_iters,
            "tol": args.tol,
            "train_frac": args.train_frac,
            "retraction": args.retraction,
            "init_gauge_condition": args.init_gauge_condition,
            "projection_dim": args.projection_dim,
            "pml_iters": args.pml_iters,
            "loss_mode": args.loss_mode,
            "loss_beta": args.loss_beta,
            "loss_eps": args.loss_eps,
            "synthetic_smoke": args.synthetic_smoke,
            "data_dir": args.data_dir,
            "device": str(device),
            "hardware": hardware_info(),
        },
    )

    summaries: list[dict[str, Any]] = []
    for seed in args.seed_values:
        setup_seed(seed)
        for k in args.ks:
            summaries.extend(run_seed_k(seed, k, args, device=device))
    write_csv(results_dir / "summary.csv", summaries)
    write_global_reports(results_dir, args.seed_values, args.ks, args.methods)


if __name__ == "__main__":
    main()
