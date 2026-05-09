from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.utils import matrix_power_symmetric, ortho


@dataclass(frozen=True)
class LowRankState:
    b: torch.Tensor
    a: torch.Tensor


def matrix_from_state(state: LowRankState) -> torch.Tensor:
    return state.b @ state.a


def factor_grams(state: LowRankState, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    s_b = state.b.transpose(-1, -2) @ state.b
    s_a = state.a @ state.a.transpose(-1, -2)
    eye = torch.eye(s_b.shape[-1], dtype=state.b.dtype, device=state.b.device)
    return s_b + eps * eye, s_a + eps * eye


def safe_invsqrt_spd(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return matrix_power_symmetric(x + eps * eye, -0.5, eps=eps)


def condition_number(x: torch.Tensor, eps: float = 1e-12) -> float:
    s = torch.linalg.svdvals(x)
    if s.numel() == 0:
        return 0.0
    return float((s.max() / s.min().clamp_min(eps)).item())


def factor_norm_sq(state: LowRankState) -> float:
    return float((state.b.pow(2).sum() + state.a.pow(2).sum()).item())


def pair_fro_norm(x_b: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(x_b * x_b) + torch.sum(x_a * x_a))


def pair_dual_norm(h_b: torch.Tensor, h_a: torch.Tensor, norm_type: str) -> float:
    if norm_type == "frobenius":
        return float(pair_fro_norm(h_b, h_a).item())
    if norm_type == "frobenius_block":
        return float(torch.linalg.norm(h_b, ord="fro").item() + torch.linalg.norm(h_a, ord="fro").item())
    if norm_type == "spectral":
        return float(torch.linalg.norm(h_b, ord="nuc").item() + torch.linalg.norm(h_a, ord="nuc").item())
    if norm_type == "nuclear":
        return float(max(torch.linalg.norm(h_b, ord=2).item(), torch.linalg.norm(h_a, ord=2).item()))
    raise ValueError(f"Unknown norm_type: {norm_type}")


def whitened_gradients(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_b, s_a = factor_grams(state)
    invsqrt_b = safe_invsqrt_spd(s_b)
    invsqrt_a = safe_invsqrt_spd(s_a)
    h_b = egrad_b @ invsqrt_a
    h_a = invsqrt_b @ egrad_a
    return h_b, h_a, invsqrt_b, invsqrt_a


def _rank1(x: torch.Tensor) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    if s.numel() == 0 or float(s[0].item()) <= 1e-14:
        return torch.zeros_like(x)
    return u[:, :1] @ vh[:1, :]


def riemannian_gd_step(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_b, h_a, invsqrt_b, invsqrt_a = whitened_gradients(state, egrad_b, egrad_a)
    denom = pair_fro_norm(h_b, h_a)
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    z_b = h_b / denom
    z_a = h_a / denom
    return z_b @ invsqrt_a, invsqrt_b @ z_a


def riemannian_gd_block_step(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_b, h_a, invsqrt_b, invsqrt_a = whitened_gradients(state, egrad_b, egrad_a)
    denom_b = torch.linalg.norm(h_b, ord="fro")
    denom_a = torch.linalg.norm(h_a, ord="fro")
    z_b = torch.zeros_like(h_b) if (not torch.isfinite(denom_b) or float(denom_b.item()) <= 1e-14) else h_b / denom_b
    z_a = torch.zeros_like(h_a) if (not torch.isfinite(denom_a) or float(denom_a.item()) <= 1e-14) else h_a / denom_a
    return z_b @ invsqrt_a, invsqrt_b @ z_a


def scaled_muon_step(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_b, h_a, invsqrt_b, invsqrt_a = whitened_gradients(state, egrad_b, egrad_a)
    if float(pair_fro_norm(h_b, h_a).item()) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    z_b = ortho(h_b)
    z_a = ortho(h_a.transpose(-1, -2)).transpose(-1, -2)
    return z_b @ invsqrt_a, invsqrt_b @ z_a


def scaled_numuon_step(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_b, h_a, invsqrt_b, invsqrt_a = whitened_gradients(state, egrad_b, egrad_a)
    if max(float(torch.linalg.norm(h_b, ord=2).item()), float(torch.linalg.norm(h_a, ord=2).item())) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    z_b = _rank1(h_b)
    z_a = _rank1(h_a)
    return z_b @ invsqrt_a, invsqrt_b @ z_a


def euclidean_gd_step(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del state
    denom = pair_fro_norm(egrad_b, egrad_a)
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-14:
        return torch.zeros_like(egrad_b), torch.zeros_like(egrad_a)
    return egrad_b / denom, egrad_a / denom


def euclidean_gd_block_step(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del state
    denom_b = torch.linalg.norm(egrad_b, ord="fro")
    denom_a = torch.linalg.norm(egrad_a, ord="fro")
    xi_b = torch.zeros_like(egrad_b) if (not torch.isfinite(denom_b) or float(denom_b.item()) <= 1e-14) else egrad_b / denom_b
    xi_a = torch.zeros_like(egrad_a) if (not torch.isfinite(denom_a) or float(denom_a.item()) <= 1e-14) else egrad_a / denom_a
    return xi_b, xi_a


def euclidean_muon_step(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del state
    if float(pair_fro_norm(egrad_b, egrad_a).item()) <= 1e-14:
        return torch.zeros_like(egrad_b), torch.zeros_like(egrad_a)
    return ortho(egrad_b), ortho(egrad_a.transpose(-1, -2)).transpose(-1, -2)


def euclidean_numuon_step(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del state
    if max(float(torch.linalg.norm(egrad_b, ord=2).item()), float(torch.linalg.norm(egrad_a, ord=2).item())) <= 1e-14:
        return torch.zeros_like(egrad_b), torch.zeros_like(egrad_a)
    return _rank1(egrad_b), _rank1(egrad_a)


METHODS: dict[str, dict[str, Any]] = {
    "riemannian_gd": {"step_fn": riemannian_gd_step, "norm": "frobenius", "geometry": "coupled"},
    "riemannian_gd_block": {"step_fn": riemannian_gd_block_step, "norm": "frobenius_block", "geometry": "coupled"},
    "scaled_muon": {"step_fn": scaled_muon_step, "norm": "spectral", "geometry": "coupled"},
    "scaled_numuon": {"step_fn": scaled_numuon_step, "norm": "nuclear", "geometry": "coupled"},
    "euclidean_gd": {"step_fn": euclidean_gd_step, "norm": "frobenius", "geometry": "euclidean"},
    "euclidean_gd_block": {"step_fn": euclidean_gd_block_step, "norm": "frobenius_block", "geometry": "euclidean"},
    "euclidean_muon": {"step_fn": euclidean_muon_step, "norm": "spectral", "geometry": "euclidean"},
    "euclidean_numuon": {"step_fn": euclidean_numuon_step, "norm": "nuclear", "geometry": "euclidean"},
}


def retract_factors(
    state: LowRankState,
    xi_b: torch.Tensor,
    xi_a: torch.Tensor,
    lr: float,
    *,
    min_sigma: float = 1e-8,
    max_backtracks: int = 12,
) -> LowRankState:
    step = float(lr)
    last = state
    for _ in range(max_backtracks + 1):
        b_new = state.b - step * xi_b
        a_new = state.a - step * xi_a
        if not torch.isfinite(b_new).all() or not torch.isfinite(a_new).all():
            step *= 0.5
            continue
        smin_b = float(torch.linalg.svdvals(b_new).min().item())
        smin_a = float(torch.linalg.svdvals(a_new).min().item())
        candidate = LowRankState(b=b_new.detach(), a=a_new.detach())
        last = candidate
        if smin_b > min_sigma and smin_a > min_sigma:
            return candidate
        step *= 0.5
    return last


def method_metrics(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
    xi_b: torch.Tensor,
    xi_a: torch.Tensor,
    norm_type: str,
    geometry: str,
) -> dict[str, float]:
    s_b, s_a = factor_grams(state)
    if geometry == "coupled":
        h_b, h_a, _, _ = whitened_gradients(state, egrad_b, egrad_a)
        z_b = xi_b @ matrix_power_symmetric(s_a, 0.5)
        z_a = matrix_power_symmetric(s_b, 0.5) @ xi_a
        rgrad_norm = float(pair_fro_norm(h_b, h_a).item())
    else:
        h_b, h_a = egrad_b, egrad_a
        z_b, z_a = xi_b, xi_a
        rgrad_norm = float(pair_fro_norm(egrad_b, egrad_a).item())
    return {
        "rgrad_norm": rgrad_norm,
        "dual_norm_H": pair_dual_norm(h_b, h_a, norm_type),
        "Z_norm_sq": float((torch.sum(z_b * z_b) + torch.sum(z_a * z_a)).item()),
        "kappa_B": condition_number(state.b),
        "kappa_A": condition_number(state.a),
        "factor_norm_sq": factor_norm_sq(state),
    }
