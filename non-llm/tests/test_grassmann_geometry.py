from __future__ import annotations

import torch

from src.manifolds.grassmann import (
    euclidean_muon_step,
    grassmann_muon_step,
    grassmann_numuon_step,
    grassmann_riemannian_gd_step,
    horizontal_projection,
    orthonormalize,
    retraction_qr,
    retraction_tall_matrix,
    whiten_tangent,
)
from src.metrics import compute_metrics_grassmann
from src.objectives.grassmann_barycenter import (
    barycenter_egrad,
    barycenter_loss,
    grassmann_distance,
    pair_loss,
)


torch.set_default_dtype(torch.float64)


def random_subspace(m: int = 8, k: int = 3) -> torch.Tensor:
    return orthonormalize(torch.randn(m, k))


def test_distance_zero_for_same_subspace() -> None:
    y = random_subspace()
    assert float(grassmann_distance(y, y).item()) < 1e-10


def test_soft_nuclear_pair_loss_zero_for_same_subspace() -> None:
    y = random_subspace()
    assert float(pair_loss(y, y, loss_mode="soft_nuclear").item()) < 1e-10


def test_qr_retraction_returns_orthonormal_basis() -> None:
    y = random_subspace()
    egrad = torch.randn_like(y)
    xi = grassmann_riemannian_gd_step(y, egrad)
    y_next = retraction_qr(y, xi, eta=0.1)
    eye = torch.eye(y.shape[1], dtype=y.dtype)
    assert torch.allclose(y_next.T @ y_next, eye, atol=1e-10)


def test_tall_retraction_keeps_full_rank_but_not_orthonormalizes() -> None:
    y = random_subspace()
    egrad = torch.randn_like(y)
    xi = grassmann_muon_step(y, egrad)
    y_next = retraction_tall_matrix(y, xi, eta=0.2)
    eigvals = torch.linalg.eigvalsh(y_next.T @ y_next)
    assert eigvals.min().item() > 1e-10
    assert torch.linalg.norm(y_next.T @ y_next - torch.eye(y.shape[1], dtype=y.dtype), ord="fro").item() > 1e-5


def test_steps_are_horizontal() -> None:
    y = random_subspace(m=10, k=4)
    egrad = torch.randn_like(y)
    for step_fn in [grassmann_riemannian_gd_step, grassmann_muon_step, grassmann_numuon_step, euclidean_muon_step]:
        xi = step_fn(y, egrad)
        assert torch.linalg.norm(y.T @ xi, ord="fro").item() < 1e-10


def test_muon_respects_effective_dimension_bound_when_rank_deficient() -> None:
    y = random_subspace(m=5, k=3)
    egrad = torch.randn_like(y)
    projected = horizontal_projection(y, egrad)
    xi = grassmann_muon_step(y, projected)
    z_norm_sq = torch.sum(whiten_tangent(y, xi) ** 2).item()
    assert z_norm_sq <= min(y.shape[0] - y.shape[1], y.shape[1]) + 1e-8


def test_barycenter_gradient_matches_finite_difference() -> None:
    torch.manual_seed(11)
    y = random_subspace(m=7, k=3)
    subspaces = torch.stack([random_subspace(m=7, k=3) for _ in range(4)])
    direction = horizontal_projection(y, torch.randn_like(y))
    eps = 1e-6
    plus = retraction_qr(y, direction, eta=-eps)
    minus = retraction_qr(y, direction, eta=eps)
    finite_diff = (barycenter_loss(plus, subspaces) - barycenter_loss(minus, subspaces)) / (2 * eps)
    inner = torch.sum(barycenter_egrad(y, subspaces) * direction)
    rel = torch.abs(finite_diff - inner) / max(1.0, abs(float(finite_diff.item())), abs(float(inner.item())))
    assert float(rel.item()) < 1e-4


def test_barycenter_gradient_matches_finite_difference_in_tall_gauge() -> None:
    torch.manual_seed(12)
    q = random_subspace(m=7, k=3)
    scale = torch.diag(torch.tensor([0.4, 1.3, 2.5], dtype=q.dtype))
    y = q @ scale
    subspaces = torch.stack([random_subspace(m=7, k=3) for _ in range(4)])
    direction = horizontal_projection(y, torch.randn_like(y))
    eps = 1e-6
    plus = y + eps * direction
    minus = y - eps * direction
    finite_diff = (barycenter_loss(plus, subspaces) - barycenter_loss(minus, subspaces)) / (2 * eps)
    inner = torch.sum(barycenter_egrad(y, subspaces) * direction)
    rel = torch.abs(finite_diff - inner) / max(1.0, abs(float(finite_diff.item())), abs(float(inner.item())))
    assert float(rel.item()) < 1e-4


def test_fro_softnuclear_gradient_matches_finite_difference_in_tall_gauge() -> None:
    torch.manual_seed(13)
    q = random_subspace(m=7, k=3)
    scale = torch.diag(torch.tensor([0.6, 1.1, 2.0], dtype=q.dtype))
    y = q @ scale
    subspaces = torch.stack([random_subspace(m=7, k=3) for _ in range(4)])
    direction = horizontal_projection(y, torch.randn_like(y))
    eps = 1e-6
    plus = y + eps * direction
    minus = y - eps * direction
    finite_diff = (
        barycenter_loss(plus, subspaces, loss_mode="fro_softnuclear")
        - barycenter_loss(minus, subspaces, loss_mode="fro_softnuclear")
    ) / (2 * eps)
    inner = torch.sum(barycenter_egrad(y, subspaces, loss_mode="fro_softnuclear") * direction)
    rel = torch.abs(finite_diff - inner) / max(1.0, abs(float(finite_diff.item())), abs(float(inner.item())))
    assert float(rel.item()) < 5e-4


def test_metrics_have_dual_norm_and_update_norms() -> None:
    y = random_subspace(m=9, k=3)
    egrad = torch.randn_like(y)
    xi = grassmann_muon_step(y, egrad)
    metrics = compute_metrics_grassmann(y, egrad, xi, norm_type="spectral")
    assert metrics["dual_norm_H"] > 0
    assert metrics["Z_norm_sq"] <= metrics["effective_dim_bound"] + 1e-8
    assert metrics["horizontal_residual"] < 1e-10
