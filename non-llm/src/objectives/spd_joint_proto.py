from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from src.objectives.spd_barycenter import arithmetic_mean, log_euclidean_mean, spd_distance_sq
from src.utils import matrix_log_symmetric, matrix_power_symmetric, sym


PairwiseFn = Callable[..., torch.Tensor]


def _batched_spd_distance_sq_to_prototypes(
    covs: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """Compute all pairwise squared AI distances from `covs` to `prototypes`.

    Returns a tensor of shape `(batch, num_prototypes)`, where entry `(i, j)`
    equals `spd_distance_sq(prototypes[j], covs[i])`.
    """
    proto_inv_half = matrix_power_symmetric(prototypes, -0.5)  # (C, d, d)
    inner = sym(proto_inv_half.unsqueeze(0) @ covs.unsqueeze(1) @ proto_inv_half.unsqueeze(0))
    log_inner = matrix_log_symmetric(inner)
    return torch.sum(log_inner * log_inner, dim=(-2, -1))


def batch_proto_scores(
    covs: torch.Tensor,
    prototypes: torch.Tensor,
    pairwise_fn: PairwiseFn,
    pairwise_kwargs: dict[str, Any] | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    pairwise_kwargs = {} if pairwise_kwargs is None else dict(pairwise_kwargs)
    if pairwise_fn is spd_distance_sq and not pairwise_kwargs:
        return -float(alpha) * _batched_spd_distance_sq_to_prototypes(covs, prototypes)
    rows = []
    for cov in covs:
        scores_i = []
        for proto in prototypes:
            dist = pairwise_fn(proto, cov, **pairwise_kwargs)
            scores_i.append(-float(alpha) * dist)
        rows.append(torch.stack(scores_i))
    return torch.stack(rows)


def joint_proto_ce_loss(
    covs: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    pairwise_fn: PairwiseFn,
    pairwise_kwargs: dict[str, Any] | None,
    alpha: float,
    reg_lambda: float,
    init_prototypes: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = labels.to(device=covs.device, dtype=torch.long)
    scores = batch_proto_scores(
        covs,
        prototypes,
        pairwise_fn=pairwise_fn,
        pairwise_kwargs=pairwise_kwargs,
        alpha=alpha,
    )
    cls_loss = F.cross_entropy(scores, labels)

    anchor_loss = torch.zeros((), dtype=prototypes.dtype, device=prototypes.device)
    if float(reg_lambda) != 0.0:
        proto_inv_half = matrix_power_symmetric(prototypes, -0.5)
        inner = sym(proto_inv_half @ init_prototypes @ proto_inv_half)
        log_inner = matrix_log_symmetric(inner)
        anchor_loss = float(reg_lambda) * torch.sum(log_inner * log_inner)

    total = cls_loss + anchor_loss
    return total, {
        "cls_loss": cls_loss.detach(),
        "anchor_loss": anchor_loss.detach(),
    }


@dataclass(frozen=True)
class JointProtoEval:
    accuracy: float
    kappa: float
    predictions: torch.Tensor
    scores: torch.Tensor


def eval_joint_proto_classifier(
    covs: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    pairwise_fn: PairwiseFn,
    pairwise_kwargs: dict[str, Any] | None = None,
    alpha: float = 1.0,
) -> JointProtoEval:
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    with torch.no_grad():
        scores = batch_proto_scores(
            covs,
            prototypes,
            pairwise_fn=pairwise_fn,
            pairwise_kwargs=pairwise_kwargs,
            alpha=alpha,
        )
        preds = scores.argmax(dim=1)
        labels_cpu = labels.detach().cpu().long()
        preds_cpu = preds.detach().cpu().long()
        accuracy = float(accuracy_score(labels_cpu.numpy(), preds_cpu.numpy()))
        kappa = float(cohen_kappa_score(labels_cpu.numpy(), preds_cpu.numpy()))
    return JointProtoEval(
        accuracy=accuracy,
        kappa=kappa,
        predictions=preds,
        scores=scores,
    )


def prediction_rows(
    labels: torch.Tensor,
    eval_result: JointProtoEval,
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


def build_initial_prototypes(
    train_covs: torch.Tensor,
    train_labels: list[str] | torch.Tensor,
    classes: list[str],
    init_method: str = "arithmetic",
) -> dict[str, torch.Tensor]:
    if init_method == "arithmetic":
        mean_fn = arithmetic_mean
    elif init_method == "log_euclidean":
        mean_fn = log_euclidean_mean
    else:
        raise ValueError(f"unknown init_method: {init_method}")

    if isinstance(train_labels, torch.Tensor):
        train_labels_list = [classes[int(idx)] for idx in train_labels.detach().cpu().tolist()]
    else:
        train_labels_list = [str(item) for item in train_labels]

    initializers: dict[str, torch.Tensor] = {}
    for cls in classes:
        mask = torch.as_tensor(
            [label == cls for label in train_labels_list],
            dtype=torch.bool,
            device=train_covs.device,
        )
        covs_c = train_covs[mask]
        initializers[cls] = mean_fn(covs_c).detach()
    return initializers
