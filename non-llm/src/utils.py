from __future__ import annotations

import csv
import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


DEFAULT_DTYPE = torch.float64


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _identity_like(x: torch.Tensor) -> torch.Tensor:
    return torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)


def _safe_eigh_symmetric(
    x: torch.Tensor,
    *,
    eps: float = 1e-12,
    clamp_min: float | None = None,
    jitter_tries: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = sym(x)
    if not torch.isfinite(y).all():
        raise ValueError("symmetric eigendecomposition received non-finite input")

    eye = _identity_like(y)
    fro_norm = torch.linalg.norm(y.detach(), ord="fro", dim=(-2, -1))
    scale = max(float(fro_norm.max().item()), 1.0)
    last_error: Exception | None = None
    for attempt in range(jitter_tries + 1):
        try:
            vals, vecs = torch.linalg.eigh(y)
            if clamp_min is not None:
                vals = vals.clamp_min(clamp_min)
            return vals, vecs
        except (RuntimeError, torch._C._LinAlgError) as exc:
            last_error = exc
            jitter = eps * (10.0**attempt) * scale
            y = sym(y + jitter * eye)

    try:
        vals, vecs = torch.linalg.eigh(y.cpu())
        vals = vals.to(device=x.device)
        vecs = vecs.to(device=x.device)
        if clamp_min is not None:
            vals = vals.clamp_min(clamp_min)
        return vals, vecs
    except (RuntimeError, torch._C._LinAlgError) as exc:
        raise RuntimeError("symmetric eigendecomposition failed after jitter and CPU fallback") from (last_error or exc)


def eigh_symmetric(x: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    return _safe_eigh_symmetric(x, eps=eps, clamp_min=eps)


def eigh_symmetric_raw(x: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    return _safe_eigh_symmetric(x, eps=eps, clamp_min=None)


def matrix_power_symmetric(x: torch.Tensor, power: float, eps: float = 1e-12) -> torch.Tensor:
    vals, vecs = eigh_symmetric(x, eps=eps)
    vals_p = vals.pow(power)
    return sym((vecs * vals_p.unsqueeze(-2)) @ vecs.transpose(-1, -2))


def matrix_log_symmetric(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    vals, vecs = eigh_symmetric(x, eps=eps)
    return sym((vecs * torch.log(vals).unsqueeze(-2)) @ vecs.transpose(-1, -2))


def matrix_exp_symmetric(x: torch.Tensor, clamp: float = 60.0) -> torch.Tensor:
    vals, vecs = _safe_eigh_symmetric(x, clamp_min=None)
    vals = vals.clamp(min=-clamp, max=clamp)
    return sym((vecs * torch.exp(vals).unsqueeze(-2)) @ vecs.transpose(-1, -2))


def matrix_sign_symmetric(x: torch.Tensor, zero_tol: float = 1e-12) -> torch.Tensor:
    y = sym(x)
    if not torch.isfinite(y).all():
        return torch.zeros_like(y)
    if float(torch.linalg.norm(y.detach(), ord="fro").item()) <= zero_tol:
        return torch.zeros_like(y)
    try:
        vals, vecs = torch.linalg.eigh(y)
    except (RuntimeError, torch._C._LinAlgError):
        u, s, vh = torch.linalg.svd(y, full_matrices=False)
        tol = max(zero_tol, float(torch.finfo(s.dtype).eps) * max(y.shape) * float(s.max().item() if s.numel() else 1.0))
        keep = s > tol
        if not bool(keep.any().item()):
            return torch.zeros_like(y)
        return sym(u[:, keep] @ vh[keep, :])
    if zero_tol > 0:
        signs = torch.where(vals.abs() <= zero_tol, torch.zeros_like(vals), torch.sign(vals))
    else:
        signs = torch.sign(vals)
    return sym((vecs * signs.unsqueeze(-2)) @ vecs.transpose(-1, -2))


def ortho(x: torch.Tensor, tol: float | None = None) -> torch.Tensor:
    """Rank-aware orthogonal polar factor.

    For rank-deficient matrices, the full SVD polar factor can fill null
    singular directions arbitrarily. Returning only the positive-singular-value
    part keeps quotient tangent constraints intact.
    """
    if x.numel() == 0:
        return torch.zeros_like(x)
    y = x
    if not torch.isfinite(y).all():
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            u, s, vh = torch.linalg.svd(y, full_matrices=False)
            break
        except (RuntimeError, torch._C._LinAlgError) as exc:
            last_error = exc
            y = y.clone()
            diag = min(y.shape[-2], y.shape[-1])
            scale = max(float(torch.linalg.norm(y.detach(), ord="fro").item()), 1.0)
            jitter = float(torch.finfo(y.dtype).eps) * (10.0**attempt) * scale
            idx = torch.arange(diag, device=y.device)
            y[..., idx, idx] = y[..., idx, idx] + jitter
    else:
        try:
            u, s, vh = torch.linalg.svd(y.cpu(), full_matrices=False)
            u = u.to(device=x.device)
            s = s.to(device=x.device)
            vh = vh.to(device=x.device)
        except (RuntimeError, torch._C._LinAlgError):
            try:
                if y.shape[-2] >= y.shape[-1]:
                    gram = y.transpose(-1, -2) @ y
                    polar = y @ matrix_power_symmetric(gram, -0.5)
                else:
                    gram = y @ y.transpose(-1, -2)
                    polar = matrix_power_symmetric(gram, -0.5) @ y
                return torch.nan_to_num(polar, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as exc:
                raise RuntimeError("ortho failed after SVD jitter, CPU SVD, and polar fallback") from (last_error or exc)
    if tol is None:
        tol = float(torch.finfo(s.dtype).eps) * max(x.shape) * float(s.max().item() if s.numel() else 1.0)
    keep = s > tol
    if not bool(keep.any().item()):
        return torch.zeros_like(x)
    return u[:, keep] @ vh[keep, :]


def project_spd(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vals, vecs = _safe_eigh_symmetric(x, eps=eps, clamp_min=eps)
    vals = vals.clamp_min(eps)
    return sym((vecs * vals.unsqueeze(-2)) @ vecs.transpose(-1, -2))


def is_spd(x: torch.Tensor, eps: float = 1e-10) -> bool:
    if not torch.isfinite(x).all():
        return False
    vals = torch.linalg.eigvalsh(sym(x))
    return bool(torch.all(vals > eps).item())


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    out = Path(path)
    ensure_dir(out.parent)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: to_jsonable(v) for k, v in row.items()} for row in rows)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        info["cpu"] = platform.processor() or platform.machine()
    return info


def format_lr(lr: float | str) -> str:
    if isinstance(lr, str):
        return lr.replace(".", "p")
    return f"{lr:.6g}".replace("-", "m").replace(".", "p")
