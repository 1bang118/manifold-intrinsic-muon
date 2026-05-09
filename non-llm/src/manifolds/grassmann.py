from __future__ import annotations

import torch

from src.utils import matrix_power_symmetric, ortho


def gram(y: torch.Tensor) -> torch.Tensor:
    return y.transpose(-1, -2) @ y


def gram_sqrt(y: torch.Tensor) -> torch.Tensor:
    return matrix_power_symmetric(gram(y), 0.5)


def gram_invsqrt(y: torch.Tensor) -> torch.Tensor:
    return matrix_power_symmetric(gram(y), -0.5)


def orthonormalize(y: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(y, mode="reduced")
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def polar_orthonormalize(y: torch.Tensor) -> torch.Tensor:
    return y @ gram_invsqrt(y)


def horizontal_projection(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    coeff = torch.linalg.solve(gram(y), y.transpose(-1, -2) @ z)
    return z - y @ coeff


def whiten_tangent(y: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    return xi @ gram_invsqrt(y)


def unwhiten_tangent(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return z @ gram_sqrt(y)


def whitened_gradient(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    return horizontal_projection(y, egrad) @ gram_sqrt(y)


def retraction_qr(y: torch.Tensor, xi: torch.Tensor, eta: float) -> torch.Tensor:
    return orthonormalize(y - float(eta) * xi)


def retraction_tall_matrix(
    y: torch.Tensor,
    xi: torch.Tensor,
    eta: float,
    min_eig: float = 1e-10,
    max_backtracks: int = 20,
) -> torch.Tensor:
    """Full-rank tall-matrix quotient retraction.

    This keeps the non-orthonormal representative instead of choosing the
    Stiefel/QR gauge. Backtracking is only to preserve full column rank.
    """
    step = float(eta)
    for _ in range(max_backtracks + 1):
        y_new = y - step * xi
        yty = gram(y_new)
        if torch.isfinite(yty).all():
            eig_min = torch.linalg.eigvalsh(yty).min()
            if bool((eig_min > min_eig).item()):
                return y_new
        step *= 0.5
    raise RuntimeError("tall-matrix Grassmann retraction could not preserve full rank")


def _safe_normalize(x: torch.Tensor, norm: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    if not torch.isfinite(norm) or float(norm.item()) <= eps:
        return torch.zeros_like(x)
    return x / norm


def grassmann_riemannian_gd_step(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = whitened_gradient(y, egrad)
    norm_h = torch.linalg.norm(h, ord="fro")
    return _safe_normalize(h @ gram_sqrt(y), norm_h)


def grassmann_muon_step(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = whitened_gradient(y, egrad)
    if float(torch.linalg.norm(h, ord="fro").item()) <= 1e-14:
        return torch.zeros_like(y)
    return ortho(h) @ gram_sqrt(y)


def grassmann_numuon_step(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    h = whitened_gradient(y, egrad)
    u, s, vh = torch.linalg.svd(h, full_matrices=False)
    if s.numel() == 0 or float(s[0].item()) <= 1e-14:
        return torch.zeros_like(y)
    return (u[:, :1] @ vh[:1, :]) @ gram_sqrt(y)


def euclidean_muon_step(y: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    projected = horizontal_projection(y, egrad)
    if float(torch.linalg.norm(projected, ord="fro").item()) <= 1e-14:
        return torch.zeros_like(y)
    return ortho(projected)
