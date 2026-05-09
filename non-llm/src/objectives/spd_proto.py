from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.objectives.spd_barycenter import arithmetic_mean as _arithmetic_mean
from src.utils import (
    eigh_symmetric,
    eigh_symmetric_raw,
    matrix_exp_symmetric,
    matrix_log_symmetric,
    matrix_power_symmetric,
    project_spd,
    sym,
)


def arithmetic_mean(covs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return _arithmetic_mean(covs, eps=eps)


def log_euclidean_mean(covs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    logs = torch.stack([matrix_log_symmetric(cov) for cov in covs])
    mean = matrix_exp_symmetric(sym(logs.mean(dim=0)))
    return project_spd(mean, eps=eps)


def spd_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_inv_half = matrix_power_symmetric(x, -0.5)
    inner = sym(x_inv_half @ sym(y) @ x_inv_half)
    log_inner = matrix_log_symmetric(inner)
    return torch.linalg.norm(log_inner, ord="fro")


def spd_distance_sq(x: torch.Tensor, y: torch.Tensor, near_identity_tol: float = 1e-10) -> torch.Tensor:
    x_inv_half = matrix_power_symmetric(x, -0.5)
    inner = sym(x_inv_half @ sym(y) @ x_inv_half)
    with torch.no_grad():
        identity = torch.eye(inner.shape[-1], dtype=inner.dtype, device=inner.device)
        near_identity = torch.linalg.norm(inner - identity, ord="fro") < near_identity_tol
    if bool(near_identity):
        delta = sym(x_inv_half @ (sym(y) - sym(x)) @ x_inv_half)
        return torch.sum(delta * delta)
    log_inner = matrix_log_symmetric(inner)
    return torch.sum(log_inner * log_inner)


def batch_spd_scores(covs: torch.Tensor, prototypes: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Return scores[i, c] = -alpha * d_AI(covs[i], prototypes[c])^2."""
    rows = []
    for cov in covs:
        scores_i = []
        for proto in prototypes:
            dist_sq = spd_distance_sq(cov, proto)
            scores_i.append(-float(alpha) * dist_sq)
        rows.append(torch.stack(scores_i))
    return torch.stack(rows)


def proto_ce_loss(
    covs: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    alpha: float,
    reg_lambda: float,
    init_prototypes: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = labels.to(device=covs.device, dtype=torch.long)
    scores = batch_spd_scores(covs, prototypes, alpha=alpha)
    cls_loss = F.cross_entropy(scores, labels)

    reg_loss = torch.zeros((), dtype=prototypes.dtype, device=prototypes.device)
    if float(reg_lambda) != 0.0:
        for proto, init_proto in zip(prototypes, init_prototypes):
            reg_loss = reg_loss + spd_distance_sq(proto, init_proto)
        reg_loss = float(reg_lambda) * reg_loss

    total = cls_loss + reg_loss
    return total, {
        "cls_loss": cls_loss.detach(),
        "anchor_loss": reg_loss.detach(),
        "reg_loss": reg_loss.detach(),
    }


def _labels_cpu(labels: torch.Tensor) -> torch.Tensor:
    return labels.detach().cpu().long()


def build_standard_episode(labels: torch.Tensor, k: int, rng: np.random.Generator) -> dict[str, list[int]]:
    """Sample k support examples per class uniformly, using all remaining trials as query."""
    labels_cpu = _labels_cpu(labels)
    support_indices: list[int] = []
    query_indices: list[int] = []
    for class_idx in sorted(labels_cpu.unique().tolist()):
        indices = (labels_cpu == int(class_idx)).nonzero(as_tuple=True)[0].numpy()
        if len(indices) <= k:
            raise ValueError(f"class {class_idx} has {len(indices)} examples, cannot reserve k={k} support examples")
        chosen = rng.choice(indices, size=k, replace=False)
        chosen_set = {int(idx) for idx in chosen.tolist()}
        support_indices.extend(sorted(chosen_set))
        query_indices.extend(int(idx) for idx in indices.tolist() if int(idx) not in chosen_set)
    support_indices.sort()
    query_indices.sort()
    return {"support_indices": support_indices, "query_indices": query_indices}


def _minmax_by_class(values: torch.Tensor, labels: torch.Tensor, eps: float) -> torch.Tensor:
    labels_cpu = _labels_cpu(labels)
    out = torch.zeros_like(values)
    for class_idx in sorted(labels_cpu.unique().tolist()):
        mask = labels.to(device=values.device, dtype=torch.long) == int(class_idx)
        class_values = values[mask]
        lo = class_values.min()
        hi = class_values.max()
        denom = (hi - lo).clamp_min(eps)
        out[mask] = (class_values - lo) / denom
    return out


def compute_hardness_scores(
    covs: torch.Tensor,
    labels: torch.Tensor,
    base_prototypes: torch.Tensor,
    eps: float = 1e-12,
    weights: dict[str, float] | None = None,
    return_components: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Compute the hard-tail score from distance, condition spread, and relative log-spectrum tail."""
    weights = {"dist": 0.5, "spread": 0.25, "tail": 0.25} if weights is None else weights
    labels_long = labels.to(device=covs.device, dtype=torch.long)
    dist_values = []
    spread_values = []
    tail_values = []

    for cov, label in zip(covs, labels_long):
        proto = base_prototypes[int(label.item())]
        dist_values.append(spd_distance_sq(cov, proto).detach())

        eigvals, _ = eigh_symmetric(cov, eps=eps)
        spread_values.append(torch.log(eigvals.max() / eigvals.min()).detach())

        proto_inv_half = matrix_power_symmetric(proto, -0.5, eps=eps)
        relative = sym(proto_inv_half @ sym(cov) @ proto_inv_half)
        log_relative = matrix_log_symmetric(relative, eps=eps)
        log_eigs, _ = eigh_symmetric_raw(log_relative)
        log_eigs = log_eigs.abs()
        nuclear = log_eigs.sum()
        fro = torch.linalg.norm(log_relative, ord="fro")
        tail_values.append((nuclear / (fro + eps)).detach())

    dist = torch.stack(dist_values).to(device=covs.device, dtype=covs.dtype)
    spread = torch.stack(spread_values).to(device=covs.device, dtype=covs.dtype)
    tail = torch.stack(tail_values).to(device=covs.device, dtype=covs.dtype)
    dist_norm = _minmax_by_class(dist, labels_long, eps)
    spread_norm = _minmax_by_class(spread, labels_long, eps)
    tail_norm = _minmax_by_class(tail, labels_long, eps)
    score = (
        float(weights.get("dist", 0.5)) * dist_norm
        + float(weights.get("spread", 0.25)) * spread_norm
        + float(weights.get("tail", 0.25)) * tail_norm
    )
    if return_components:
        return {
            "hardness": score,
            "dist": dist,
            "spread": spread,
            "tail": tail,
            "dist_norm": dist_norm,
            "spread_norm": spread_norm,
            "tail_norm": tail_norm,
        }
    return score


def build_hardtail_episode(
    labels: torch.Tensor,
    hardness_scores: torch.Tensor,
    k: int,
    tail_fraction: float,
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    """Sample support examples uniformly from the top hardness tail within each class."""
    labels_cpu = _labels_cpu(labels)
    hardness_cpu = hardness_scores.detach().cpu()
    support_indices: list[int] = []
    query_indices: list[int] = []
    for class_idx in sorted(labels_cpu.unique().tolist()):
        indices = (labels_cpu == int(class_idx)).nonzero(as_tuple=True)[0].numpy()
        if len(indices) <= k:
            raise ValueError(f"class {class_idx} has {len(indices)} examples, cannot reserve k={k} support examples")
        pool_size = max(k, int(np.ceil(float(tail_fraction) * len(indices))))
        order = sorted(indices.tolist(), key=lambda idx: float(hardness_cpu[int(idx)].item()), reverse=True)
        pool = np.asarray(order[:pool_size], dtype=np.int64)
        chosen = rng.choice(pool, size=k, replace=False)
        chosen_set = {int(idx) for idx in chosen.tolist()}
        support_indices.extend(sorted(chosen_set))
        query_indices.extend(int(idx) for idx in indices.tolist() if int(idx) not in chosen_set)
    support_indices.sort()
    query_indices.sort()
    return {"support_indices": support_indices, "query_indices": query_indices}


@dataclass(frozen=True)
class ProtoEval:
    accuracy: float
    kappa: float
    predictions: torch.Tensor
    scores: torch.Tensor


def eval_proto_classifier(
    covs: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    alpha: float = 1.0,
) -> ProtoEval:
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    with torch.no_grad():
        scores = batch_spd_scores(covs, prototypes, alpha=alpha)
        preds = scores.argmax(dim=1)
        labels_cpu = labels.detach().cpu().long()
        preds_cpu = preds.detach().cpu().long()
        accuracy = float(accuracy_score(labels_cpu.numpy(), preds_cpu.numpy()))
        kappa = float(cohen_kappa_score(labels_cpu.numpy(), preds_cpu.numpy()))
    return ProtoEval(accuracy=accuracy, kappa=kappa, predictions=preds, scores=scores)


def prediction_rows(
    labels: torch.Tensor,
    eval_result: ProtoEval,
    class_names: list[str],
    ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    ids = list(range(int(labels.numel()))) if ids is None else ids
    rows: list[dict[str, Any]] = []
    labels_cpu = labels.detach().cpu().long()
    preds_cpu = eval_result.predictions.detach().cpu().long()
    scores_cpu = eval_result.scores.detach().cpu()
    for row_id, label, pred, scores in zip(ids, labels_cpu, preds_cpu, scores_cpu):
        row: dict[str, Any] = {
            "trial_id": int(row_id),
            "true_label": class_names[int(label)],
            "predicted_label": class_names[int(pred)],
        }
        for class_idx, class_name in enumerate(class_names):
            row[f"score_{class_name}"] = float(scores[class_idx].item())
        rows.append(row)
    return rows


def split_session2_support_query(
    covs: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    seed: int,
) -> dict[str, torch.Tensor | list[int]]:
    labels = labels.detach().cpu().long()
    rng = np.random.default_rng(seed)
    support_indices: list[int] = []
    query_indices: list[int] = []
    for class_idx in sorted(labels.unique().tolist()):
        indices = (labels == int(class_idx)).nonzero(as_tuple=True)[0].numpy()
        if len(indices) <= k:
            raise ValueError(f"class {class_idx} has {len(indices)} examples, cannot reserve k={k} support examples")
        rng.shuffle(indices)
        support_indices.extend(int(i) for i in sorted(indices[:k]))
        query_indices.extend(int(i) for i in sorted(indices[k:]))
    support_indices.sort()
    query_indices.sort()
    support = torch.as_tensor(support_indices, dtype=torch.long, device=covs.device)
    query = torch.as_tensor(query_indices, dtype=torch.long, device=covs.device)
    return {
        "support_covs": covs[support],
        "support_labels": labels.to(covs.device)[support],
        "query_covs": covs[query],
        "query_labels": labels.to(covs.device)[query],
        "support_indices": support_indices,
        "query_indices": query_indices,
    }
