from __future__ import annotations

import torch

from src.objectives.spd_proto import (
    batch_spd_scores,
    build_hardtail_episode,
    build_standard_episode,
    compute_hardness_scores,
    eval_proto_classifier,
    log_euclidean_mean,
    proto_ce_loss,
    split_session2_support_query,
)
from src.utils import project_spd, sym


torch.set_default_dtype(torch.float64)


def random_spd(n: int = 5) -> torch.Tensor:
    a = torch.randn(n, n)
    return project_spd(sym(a @ a.T) + 0.5 * torch.eye(n, dtype=a.dtype))


def test_proto_scores_and_loss_have_gradients() -> None:
    covs = torch.stack([random_spd() for _ in range(6)])
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    prototypes = torch.stack([log_euclidean_mean(covs[labels == c]) for c in [0, 1]]).requires_grad_(True)
    scores = batch_spd_scores(covs, prototypes, alpha=1.0)
    assert scores.shape == (6, 2)
    loss, aux = proto_ce_loss(covs, labels, prototypes, 1.0, 1e-3, prototypes.detach())
    assert loss.ndim == 0
    assert aux["cls_loss"].item() >= 0
    loss.backward()
    assert prototypes.grad is not None
    assert torch.isfinite(prototypes.grad).all()


def test_eval_and_support_query_split() -> None:
    covs = torch.stack([random_spd() for _ in range(12)])
    labels = torch.tensor([0] * 6 + [1] * 6, dtype=torch.long)
    prototypes = torch.stack([log_euclidean_mean(covs[labels == c]) for c in [0, 1]])
    result = eval_proto_classifier(covs, labels, prototypes)
    assert 0.0 <= result.accuracy <= 1.0
    split = split_session2_support_query(covs, labels, k=2, seed=0)
    assert int(split["support_labels"].numel()) == 4
    assert int(split["query_labels"].numel()) == 8


def test_episodic_samplers_and_hardness_scores_are_finite() -> None:
    covs = torch.stack([random_spd() for _ in range(16)])
    labels = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    prototypes = torch.stack([log_euclidean_mean(covs[labels == c]) for c in [0, 1]])
    components = compute_hardness_scores(covs, labels, prototypes, return_components=True)
    assert isinstance(components, dict)
    hardness = components["hardness"]
    assert hardness.shape == (16,)
    assert torch.isfinite(hardness).all()

    rng = __import__("numpy").random.default_rng(0)
    standard = build_standard_episode(labels, k=2, rng=rng)
    hardtail = build_hardtail_episode(labels, hardness, k=2, tail_fraction=0.5, rng=rng)

    for episode in [standard, hardtail]:
        assert len(episode["support_indices"]) == 4
        assert len(episode["query_indices"]) == 12
        assert not set(episode["support_indices"]).intersection(episode["query_indices"])
