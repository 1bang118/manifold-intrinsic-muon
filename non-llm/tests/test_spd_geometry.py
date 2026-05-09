from __future__ import annotations

import torch

from src.manifolds.spd import (
    euclidean_muon_step,
    euclidean_numuon_step,
    retraction_exp,
    spd_muon_step,
    spd_numuon_step,
    spd_riemannian_gd_step,
    whiten_tangent,
)
from src.metrics import compute_metrics_spd
from src.objectives.spd_barycenter import (
    arithmetic_mean,
    barycenter_egrad,
    barycenter_loss,
    log_euclidean_mean,
    spd_distance,
)
from src.utils import matrix_log_symmetric, matrix_power_symmetric, matrix_sign_symmetric, project_spd, sym


torch.set_default_dtype(torch.float64)


def make_spd(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=gen)
    return project_spd(a @ a.T + 0.5 * torch.eye(n, dtype=a.dtype))


def test_matrix_helpers_preserve_symmetry() -> None:
    x = make_spd(5, seed=0)
    for fn in [
        lambda z: matrix_power_symmetric(z, 0.5),
        lambda z: matrix_power_symmetric(z, -0.5),
        matrix_log_symmetric,
    ]:
        y = fn(x)
        assert torch.allclose(y, y.T, atol=1e-10)
        assert torch.isfinite(y).all()

    sign = matrix_sign_symmetric(torch.randn(5, 5))
    assert torch.allclose(sign, sign.T, atol=1e-10)


def test_exp_retraction_remains_spd() -> None:
    x = make_spd(4, seed=1)
    egrad = sym(torch.randn(4, 4))
    xi = spd_muon_step(x, egrad)
    x_next = retraction_exp(x, xi, eta=0.1)
    eigvals = torch.linalg.eigvalsh(x_next)
    assert torch.all(eigvals > 0)
    assert torch.allclose(x_next, x_next.T, atol=1e-10)


def test_spd_distance_zero_on_identical_matrix() -> None:
    x = make_spd(4, seed=2)
    assert spd_distance(x, x).item() < 1e-8


def test_closed_form_means_return_spd() -> None:
    covs = torch.stack([make_spd(4, seed=i) for i in range(3, 8)])
    for mean in [arithmetic_mean(covs), log_euclidean_mean(covs)]:
        assert torch.allclose(mean, mean.T, atol=1e-10)
        assert torch.all(torch.linalg.eigvalsh(mean) > 0)


def test_barycenter_gradient_and_metrics_are_finite() -> None:
    covs = torch.stack([make_spd(4, seed=i) for i in range(10, 14)])
    x = arithmetic_mean(covs)
    loss = barycenter_loss(x, covs)
    egrad = barycenter_egrad(x, covs)
    xi = spd_riemannian_gd_step(x, egrad)
    metrics = compute_metrics_spd(x, egrad, xi, norm_type="frobenius")

    assert torch.isfinite(loss)
    assert torch.isfinite(egrad).all()
    assert torch.isfinite(xi).all()
    assert metrics["rgrad_norm"] >= 0
    assert metrics["dual_norm_H"] >= 0
    assert metrics["Z_norm_sq"] >= 0


def test_muon_and_numuon_directions_have_expected_whitened_norms() -> None:
    x = make_spd(5, seed=20)
    egrad = sym(torch.randn(5, 5, generator=torch.Generator().manual_seed(21)))

    z_muon = whiten_tangent(x, spd_muon_step(x, egrad))
    z_numuon = whiten_tangent(x, spd_numuon_step(x, egrad))

    assert torch.linalg.norm(z_muon, ord="fro").item() <= (5 ** 0.5) + 1e-8
    assert torch.linalg.norm(z_numuon, ord="fro").item() <= 1.0 + 1e-8


def test_euclidean_muon_and_numuon_steps_are_symmetric_and_retractable() -> None:
    x = make_spd(5, seed=30)
    egrad = sym(torch.randn(5, 5, generator=torch.Generator().manual_seed(31)))

    xi_muon = euclidean_muon_step(x, egrad)
    xi_numuon = euclidean_numuon_step(x, egrad)

    assert torch.allclose(xi_muon, xi_muon.T, atol=1e-10)
    assert torch.allclose(xi_numuon, xi_numuon.T, atol=1e-10)

    x_muon = retraction_exp(x, xi_muon, eta=0.05)
    x_numuon = retraction_exp(x, xi_numuon, eta=0.05)

    assert torch.all(torch.linalg.eigvalsh(x_muon) > 0)
    assert torch.all(torch.linalg.eigvalsh(x_numuon) > 0)
