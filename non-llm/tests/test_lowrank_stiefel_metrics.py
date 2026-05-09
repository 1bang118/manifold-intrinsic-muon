from __future__ import annotations

import torch

from src.manifolds.lowrank import LowRankState, scaled_muon_step
from src.manifolds.stiefel import imuon_direction, stiefel_polar
from src.metrics import compute_metrics_lowrank, compute_metrics_stiefel


torch.set_default_dtype(torch.float64)


def test_lowrank_metrics_are_finite_for_coupled_spectral_step() -> None:
    generator = torch.Generator().manual_seed(101)
    state = LowRankState(
        b=torch.randn(8, 3, generator=generator),
        a=torch.randn(3, 7, generator=generator),
    )
    egrad_b = torch.randn(state.b.shape, dtype=state.b.dtype, generator=generator)
    egrad_a = torch.randn(state.a.shape, dtype=state.a.dtype, generator=generator)
    xi_b, xi_a = scaled_muon_step(state, egrad_b, egrad_a)

    metrics = compute_metrics_lowrank(state, egrad_b, egrad_a, xi_b, xi_a, norm_type="spectral", geometry="coupled")

    for value in metrics.values():
        assert torch.isfinite(torch.tensor(value))
    assert metrics["dual_norm_H"] >= 0
    assert metrics["Z_norm_sq"] >= 0


def test_stiefel_metrics_are_finite_and_direction_is_tangent() -> None:
    generator = torch.Generator().manual_seed(202)
    x = stiefel_polar(torch.randn(10, 4, generator=generator))
    egrad = torch.randn(x.shape, dtype=x.dtype, generator=generator)
    xi = imuon_direction(x, egrad)

    metrics = compute_metrics_stiefel(x, egrad, xi, method="imuon")

    for value in metrics.values():
        assert torch.isfinite(torch.tensor(value))
    assert metrics["direction_tangent_violation"] < 1e-10
    assert metrics["direction_fro"] >= 0
