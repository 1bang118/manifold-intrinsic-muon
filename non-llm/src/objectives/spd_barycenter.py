from __future__ import annotations

import torch

from src.utils import matrix_log_symmetric, matrix_power_symmetric, project_spd, sym


def spd_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_inv_half = matrix_power_symmetric(x, -0.5)
    inner = sym(x_inv_half @ sym(y) @ x_inv_half)
    log_inner = matrix_log_symmetric(inner)
    return torch.linalg.norm(log_inner, ord="fro")


def spd_distance_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_inv_half = matrix_power_symmetric(x, -0.5)
    inner = sym(x_inv_half @ sym(y) @ x_inv_half)
    log_inner = matrix_log_symmetric(inner)
    return torch.sum(log_inner * log_inner)


def barycenter_loss(x: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), dtype=x.dtype, device=x.device)
    x_inv_half = matrix_power_symmetric(x, -0.5)
    for cov in covariances:
        inner = sym(x_inv_half @ sym(cov) @ x_inv_half)
        log_inner = matrix_log_symmetric(inner)
        total = total + torch.sum(log_inner * log_inner)
    return total


def barycenter_egrad(x: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:
    """Euclidean gradient of sum of squared affine-invariant distances.

    The affine-invariant Riemannian gradient is
    ``-2 * sum Log_x(C_i)``. Since the AI metric satisfies
    ``grad_R = x @ egrad @ x``, the corresponding Euclidean gradient is
    ``-2 * x^{-1/2} log(x^{-1/2} C_i x^{-1/2}) x^{-1/2}``.
    """
    x_inv_half = matrix_power_symmetric(x, -0.5)
    grad = torch.zeros_like(x)
    for cov in covariances:
        inner = sym(x_inv_half @ sym(cov) @ x_inv_half)
        log_inner = matrix_log_symmetric(inner)
        grad = grad - 2.0 * (x_inv_half @ log_inner @ x_inv_half)
    return sym(grad)


def arithmetic_mean(covariances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return project_spd(sym(covariances.mean(dim=0)), eps=eps)


def log_euclidean_mean(covariances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    logs = torch.stack([matrix_log_symmetric(cov) for cov in covariances])
    vals, vecs = torch.linalg.eigh(sym(logs.mean(dim=0)))
    mean = sym((vecs * torch.exp(vals).unsqueeze(-2)) @ vecs.transpose(-1, -2))
    return project_spd(mean, eps=eps)
