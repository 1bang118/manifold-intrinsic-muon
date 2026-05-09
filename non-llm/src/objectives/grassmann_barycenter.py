from __future__ import annotations

import torch

from src.manifolds.grassmann import gram_invsqrt, orthonormalize, polar_orthonormalize


LOSS_MODES = {"squared_geodesic", "soft_nuclear", "fro_softnuclear"}


def _clamp_principal_singular_values(svdvals: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
    svdvals = svdvals.clamp(0.0, 1.0)
    return torch.where(svdvals > 1.0 - tol, torch.ones_like(svdvals), svdvals)


def principal_angles(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    q1 = orthonormalize(y1)
    q2 = orthonormalize(y2)
    svdvals = _clamp_principal_singular_values(torch.linalg.svdvals(q1.transpose(-1, -2) @ q2))
    return torch.acos(svdvals)


def grassmann_distance(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    theta = principal_angles(y1, y2)
    return torch.linalg.norm(theta)


def pair_loss_from_principal_angles(
    theta: torch.Tensor,
    loss_mode: str = "squared_geodesic",
    loss_beta: float = 0.25,
    loss_eps: float = 1e-6,
) -> torch.Tensor:
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"unknown Grassmann loss_mode: {loss_mode}")
    if loss_mode == "squared_geodesic":
        return torch.sum(theta * theta)
    soft_abs = torch.sqrt(theta * theta + float(loss_eps) ** 2) - float(loss_eps)
    if loss_mode == "soft_nuclear":
        return torch.sum(soft_abs)
    return float(loss_beta) * torch.sum(theta * theta) + (1.0 - float(loss_beta)) * torch.sum(soft_abs)


def pair_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    loss_mode: str = "squared_geodesic",
    loss_beta: float = 0.25,
    loss_eps: float = 1e-6,
) -> torch.Tensor:
    theta = principal_angles(y, target)
    return pair_loss_from_principal_angles(theta, loss_mode=loss_mode, loss_beta=loss_beta, loss_eps=loss_eps)


def barycenter_loss(
    y: torch.Tensor,
    subspaces: torch.Tensor | list[torch.Tensor],
    loss_mode: str = "squared_geodesic",
    loss_beta: float = 0.25,
    loss_eps: float = 1e-6,
) -> torch.Tensor:
    total = torch.zeros((), dtype=y.dtype, device=y.device)
    for subspace in subspaces:
        total = total + pair_loss(y, subspace, loss_mode=loss_mode, loss_beta=loss_beta, loss_eps=loss_eps)
    return total


def grassmann_log_map(y: torch.Tensor, target: torch.Tensor, pinv_rcond: float = 1e-10) -> torch.Tensor:
    q = orthonormalize(y)
    q_target = orthonormalize(target)
    p_perp_target = q_target - q @ (q.transpose(-1, -2) @ q_target)
    align = q.transpose(-1, -2) @ q_target
    try:
        right = torch.linalg.inv(align)
    except RuntimeError:
        right = torch.linalg.pinv(align, rtol=pinv_rcond)
    m = p_perp_target @ right
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    return u @ torch.diag(torch.atan(s)) @ vh


def _barycenter_egrad_autograd(
    y: torch.Tensor,
    subspaces: torch.Tensor | list[torch.Tensor],
    loss_mode: str,
    loss_beta: float,
    loss_eps: float,
) -> torch.Tensor:
    y_var = y.detach().clone().requires_grad_(True)
    loss = barycenter_loss(y_var, subspaces, loss_mode=loss_mode, loss_beta=loss_beta, loss_eps=loss_eps)
    (grad,) = torch.autograd.grad(loss, y_var)
    return grad.detach()


def barycenter_egrad(
    y: torch.Tensor,
    subspaces: torch.Tensor | list[torch.Tensor],
    loss_mode: str = "squared_geodesic",
    loss_beta: float = 0.25,
    loss_eps: float = 1e-6,
) -> torch.Tensor:
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"unknown Grassmann loss_mode: {loss_mode}")
    if loss_mode != "squared_geodesic":
        return _barycenter_egrad_autograd(
            y,
            subspaces,
            loss_mode=loss_mode,
            loss_beta=loss_beta,
            loss_eps=loss_eps,
        )
    q = polar_orthonormalize(y)
    grad = torch.zeros_like(q)
    for subspace in subspaces:
        grad = grad - 2.0 * grassmann_log_map(q, subspace)
    return grad @ gram_invsqrt(y)


def extrinsic_projector_mean(subspaces: torch.Tensor | list[torch.Tensor], k: int | None = None) -> torch.Tensor:
    first = subspaces[0]
    k = first.shape[1] if k is None else k
    projector_mean = torch.zeros(first.shape[0], first.shape[0], dtype=first.dtype, device=first.device)
    for subspace in subspaces:
        q = orthonormalize(subspace)
        projector_mean = projector_mean + q @ q.transpose(-1, -2)
    projector_mean = projector_mean / len(subspaces)
    eigvals, eigvecs = torch.linalg.eigh(0.5 * (projector_mean + projector_mean.transpose(-1, -2)))
    del eigvals
    return eigvecs[:, -k:]
