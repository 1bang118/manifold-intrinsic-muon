from __future__ import annotations

import torch

from src.utils import (
    eigh_symmetric_raw,
    matrix_exp_symmetric,
    matrix_power_symmetric,
    matrix_sign_symmetric,
    sym,
)


def spd_sqrt(x: torch.Tensor) -> torch.Tensor:
    return matrix_power_symmetric(x, 0.5)


def spd_invsqrt(x: torch.Tensor) -> torch.Tensor:
    return matrix_power_symmetric(x, -0.5)


def whiten_tangent(x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    x_inv_half = spd_invsqrt(x)
    return sym(x_inv_half @ sym(xi) @ x_inv_half)


def unwhiten_tangent(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    x_half = spd_sqrt(x)
    return sym(x_half @ sym(z) @ x_half)


def riemannian_gradient(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    return sym(x @ sym(egrad) @ x)


def retraction_exp(x: torch.Tensor, xi: torch.Tensor, eta: float) -> torch.Tensor:
    """Affine-invariant exponential retraction for a descent direction.

    Step functions return the positive first-order direction ``xi``. This
    retraction applies ``-eta * xi``.
    """
    x_half = spd_sqrt(x)
    inner = -float(eta) * whiten_tangent(x, xi)
    return sym(x_half @ matrix_exp_symmetric(inner) @ x_half)


def _safe_normalize(x: torch.Tensor, norm: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    if not torch.isfinite(norm) or float(norm.item()) <= eps:
        return torch.zeros_like(x)
    return x / norm


def spd_riemannian_gd_step(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    grad = riemannian_gradient(x, egrad)
    h = whiten_tangent(x, grad)
    return _safe_normalize(grad, torch.linalg.norm(h, ord="fro"))


def spd_muon_step(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = unwhiten_tangent(x, sym(egrad))
    sign_h = matrix_sign_symmetric(h)
    return unwhiten_tangent(x, sign_h)


def spd_numuon_step(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(x).all() or not torch.isfinite(egrad).all():
        return torch.zeros_like(x)
    try:
        x_half = spd_sqrt(x)
    except (RuntimeError, ValueError):
        return torch.zeros_like(x)
    if not torch.isfinite(x_half).all():
        return torch.zeros_like(x)
    h = sym(x_half @ sym(egrad) @ x_half)
    if not torch.isfinite(h).all():
        return torch.zeros_like(x)
    try:
        eigvals, eigvecs = eigh_symmetric_raw(h)
    except (RuntimeError, ValueError):
        return torch.zeros_like(x)
    idx = torch.argmax(eigvals.abs())
    sign_lambda = torch.sign(eigvals[idx])
    if float(sign_lambda.item()) == 0.0:
        return torch.zeros_like(x)
    q = eigvecs[:, idx : idx + 1]
    xi = sym(sign_lambda * (x_half @ q @ q.transpose(-1, -2) @ x_half))
    if not torch.isfinite(xi).all():
        return torch.zeros_like(x)
    return xi


def euclidean_gd_step(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    del x
    eg = sym(egrad)
    return _safe_normalize(eg, torch.linalg.norm(eg, ord="fro"))


def euclidean_muon_step(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    del x
    return matrix_sign_symmetric(sym(egrad))


def euclidean_numuon_step(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    del x
    eg = sym(egrad)
    if not torch.isfinite(eg).all():
        return torch.zeros_like(eg)
    try:
        eigvals, eigvecs = eigh_symmetric_raw(eg)
    except (RuntimeError, ValueError):
        return torch.zeros_like(eg)
    idx = torch.argmax(eigvals.abs())
    sign_lambda = torch.sign(eigvals[idx])
    if float(sign_lambda.item()) == 0.0:
        return torch.zeros_like(eg)
    q = eigvecs[:, idx : idx + 1]
    return sym(sign_lambda * (q @ q.transpose(-1, -2)))
