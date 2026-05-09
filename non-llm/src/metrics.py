from __future__ import annotations

import torch

from src.manifolds.grassmann import gram, horizontal_projection, whiten_tangent as grassmann_whiten_tangent
from src.manifolds.lowrank import (
    LowRankState,
    condition_number,
    factor_grams,
    factor_norm_sq,
    pair_dual_norm,
    pair_fro_norm,
    whitened_gradients,
)
from src.manifolds.spd import riemannian_gradient, whiten_tangent
from src.manifolds.stiefel import normal_projection, skew, spel_direction, tangent_projection
from src.utils import matrix_power_symmetric, sym


def compute_metrics_lowrank(
    state: LowRankState,
    egrad_b: torch.Tensor,
    egrad_a: torch.Tensor,
    xi_b: torch.Tensor,
    xi_a: torch.Tensor,
    norm_type: str,
    geometry: str,
) -> dict[str, float]:
    gram_b, gram_a = factor_grams(state)
    if geometry == "coupled":
        h_b, h_a, _, _ = whitened_gradients(state, egrad_b, egrad_a)
        z_b = xi_b @ matrix_power_symmetric(gram_a, 0.5)
        z_a = matrix_power_symmetric(gram_b, 0.5) @ xi_a
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


def compute_metrics_spd(
    x: torch.Tensor,
    egrad: torch.Tensor,
    xi: torch.Tensor,
    norm_type: str = "spectral",
) -> dict[str, float]:
    x_half = matrix_power_symmetric(x, 0.5)
    h = sym(x_half @ sym(egrad) @ x_half)
    rgrad = riemannian_gradient(x, egrad)
    z = whiten_tangent(x, xi)

    metrics: dict[str, float] = {
        "rgrad_norm": float(torch.linalg.norm(whiten_tangent(x, rgrad), ord="fro").item()),
        "Z_norm_sq": float(torch.sum(z * z).item()),
    }

    if norm_type == "spectral":
        metrics["dual_norm_H"] = float(torch.linalg.eigvalsh(h).abs().sum().item())
    elif norm_type == "frobenius":
        metrics["dual_norm_H"] = float(torch.linalg.norm(h, ord="fro").item())
    elif norm_type == "nuclear":
        metrics["dual_norm_H"] = float(torch.linalg.eigvalsh(h).abs().max().item())
    else:
        raise ValueError(f"unknown SPD norm_type: {norm_type}")

    eigvals = torch.linalg.eigvalsh(sym(x))
    metrics["lambda_min"] = float(eigvals.min().item())
    metrics["lambda_max"] = float(eigvals.max().item())
    metrics["condition_number"] = float((eigvals.max() / eigvals.min().clamp_min(1e-30)).item())
    return metrics


def compute_metrics_stiefel(
    x: torch.Tensor,
    egrad: torch.Tensor,
    xi: torch.Tensor,
    method: str = "",
) -> dict[str, float]:
    rgrad = tangent_projection(x, egrad)
    skew_grad = skew(x.transpose(-1, -2) @ egrad)
    normal_grad = normal_projection(x, egrad)
    skew_xi = skew(x.transpose(-1, -2) @ xi)
    normal_xi = normal_projection(x, xi)
    tangent_residual = x.transpose(-1, -2) @ xi + xi.transpose(-1, -2) @ x

    spel_xi = spel_direction(x, egrad)
    spel_residual = x.transpose(-1, -2) @ spel_xi + spel_xi.transpose(-1, -2) @ x
    normal_grad_nuc = float(torch.linalg.norm(normal_grad, ord="nuc").item())

    return {
        "rgrad_fro": float(torch.linalg.norm(rgrad, ord="fro").item()),
        "skew_grad_fro": float(torch.linalg.norm(skew_grad, ord="fro").item()),
        "normal_grad_fro": float(torch.linalg.norm(normal_grad, ord="fro").item()),
        "skew_grad_nuc": float(torch.linalg.norm(skew_grad, ord="nuc").item()),
        "normal_grad_nuc": normal_grad_nuc,
        "skew_normal_nuc_ratio": float(torch.linalg.norm(skew_grad, ord="nuc").item() / max(normal_grad_nuc, 1e-12)),
        "direction_fro": float(torch.linalg.norm(xi, ord="fro").item()),
        "direction_spectral": float(torch.linalg.norm(xi, ord=2).item()),
        "direction_tangent_violation": float(torch.linalg.norm(tangent_residual, ord="fro").item()),
        "direction_skew_fro": float(torch.linalg.norm(skew_xi, ord="fro").item()),
        "direction_normal_fro": float(torch.linalg.norm(normal_xi, ord="fro").item()),
        "spel_direction_tangent_violation": float(torch.linalg.norm(spel_residual, ord="fro").item()),
        "is_spel": 1.0 if method == "spel" else 0.0,
    }


def compute_metrics_grassmann(
    y: torch.Tensor,
    egrad: torch.Tensor,
    xi: torch.Tensor,
    norm_type: str = "spectral",
) -> dict[str, float]:
    yty = gram(y)
    yty_half = matrix_power_symmetric(yty, 0.5)
    h = horizontal_projection(y, egrad) @ yty_half
    z = grassmann_whiten_tangent(y, xi)
    svdvals_h = torch.linalg.svdvals(h)
    gram_eigs = torch.linalg.eigvalsh(yty)
    horizontal_residual = torch.linalg.norm(y.transpose(-1, -2) @ xi, ord="fro")

    metrics: dict[str, float] = {
        "rgrad_norm": float(torch.linalg.norm(h, ord="fro").item()),
        "Z_norm_sq": float(torch.sum(z * z).item()),
        "horizontal_residual": float(horizontal_residual.item()),
        "gram_min_eig": float(gram_eigs.min().item()),
        "gram_max_eig": float(gram_eigs.max().item()),
        "gram_condition_number": float((gram_eigs.max() / gram_eigs.min().clamp_min(1e-30)).item()),
        "effective_dim_bound": float(min(y.shape[0] - y.shape[1], y.shape[1])),
    }

    if norm_type == "spectral":
        metrics["dual_norm_H"] = float(svdvals_h.sum().item())
    elif norm_type == "frobenius":
        metrics["dual_norm_H"] = float(torch.linalg.norm(h, ord="fro").item())
    elif norm_type == "nuclear":
        metrics["dual_norm_H"] = float(svdvals_h.max().item() if svdvals_h.numel() else 0.0)
    else:
        raise ValueError(f"unknown Grassmann norm_type: {norm_type}")
    return metrics
