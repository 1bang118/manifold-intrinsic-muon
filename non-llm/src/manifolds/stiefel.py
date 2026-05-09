from __future__ import annotations

import torch

from src.utils import matrix_power_symmetric, ortho, sym


def skew(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x - x.transpose(-1, -2))


def stiefel_polar(y: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    gram = y.transpose(-1, -2) @ y
    return y @ matrix_power_symmetric(gram, -0.5, eps=eps)


def stiefel_qr(y: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(y, mode="reduced")
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def normal_projection(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return z - x @ (x.transpose(-1, -2) @ z)


def tangent_projection(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return z - x @ sym(x.transpose(-1, -2) @ z)


def safe_normalize(z: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    norm = torch.linalg.norm(z, ord="fro")
    if not torch.isfinite(norm) or float(norm.item()) <= eps:
        return torch.zeros_like(z)
    return z / norm


def rgd_direction(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    return safe_normalize(tangent_projection(x, egrad))


def imuon_direction(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    skew_block = skew(x.transpose(-1, -2) @ egrad)
    normal_block = normal_projection(x, egrad)
    if float(torch.linalg.norm(skew_block, ord="fro").item()) <= 1e-14 and float(torch.linalg.norm(normal_block, ord="fro").item()) <= 1e-14:
        return torch.zeros_like(x)
    return x @ skew(ortho(skew_block)) + ortho(normal_block)


def spel_direction(x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    rgrad = tangent_projection(x, egrad)
    if float(torch.linalg.norm(rgrad, ord="fro").item()) <= 1e-14:
        return torch.zeros_like(x)
    return ortho(rgrad)
