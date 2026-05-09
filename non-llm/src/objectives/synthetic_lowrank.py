from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.manifolds.lowrank import LowRankState, matrix_from_state
from src.utils import DEFAULT_DTYPE


@dataclass(frozen=True)
class SyntheticProblem:
    variant: str
    m: int
    n: int
    r_star: int
    kappa: float
    snr_db: float
    w_star: torch.Tensor
    noisy_target: torch.Tensor | None = None
    rows: torch.Tensor | None = None
    cols: torch.Tensor | None = None
    values: torch.Tensor | None = None
    sensing_ops: torch.Tensor | None = None
    sensing_targets: torch.Tensor | None = None
    sensing_seed: int | None = None
    sensing_chunk_size: int = 256
    sensing_num_meas: int | None = None


def generate_ground_truth(
    *,
    m: int,
    n: int,
    r_star: int,
    kappa: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    u_raw = torch.randn((m, r_star), generator=g, device=device, dtype=dtype)
    v_raw = torch.randn((n, r_star), generator=g, device=device, dtype=dtype)
    u_star, _ = torch.linalg.qr(u_raw)
    v_star, _ = torch.linalg.qr(v_raw)
    if r_star == 1:
        sigmas = torch.tensor([1.0], device=device, dtype=dtype)
    else:
        powers = torch.arange(r_star, device=device, dtype=dtype) / float(r_star - 1)
        sigmas = torch.tensor(float(kappa), device=device, dtype=dtype).pow(-powers)
    w_star = u_star @ torch.diag(sigmas) @ v_star.transpose(-1, -2)
    w_star = w_star / torch.linalg.norm(w_star, ord="fro").clamp_min(1e-30)
    return w_star, u_star, sigmas, v_star


def noise_std_from_snr_db(snr_db: float, m: int, n: int) -> float:
    if math.isinf(snr_db):
        return 0.0
    target_fro = 10.0 ** (-float(snr_db) / 20.0)
    return target_fro / math.sqrt(m * n)


def build_problem(
    *,
    variant: str,
    m: int,
    n: int,
    r_star: int,
    kappa: float,
    snr_db: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = DEFAULT_DTYPE,
    sensing_multiplier: float = 5.0,
    completion_multiplier: float = 5.0,
    sensing_chunk_size: int = 256,
) -> SyntheticProblem:
    w_star, _, _, _ = generate_ground_truth(
        m=m,
        n=n,
        r_star=r_star,
        kappa=kappa,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    sigma = noise_std_from_snr_db(snr_db, m, n)
    g = torch.Generator(device=device).manual_seed(seed + 10_000)

    if variant == "full":
        noise = torch.randn((m, n), generator=g, device=device, dtype=dtype) * sigma
        return SyntheticProblem(
            variant=variant,
            m=m,
            n=n,
            r_star=r_star,
            kappa=kappa,
            snr_db=snr_db,
            w_star=w_star,
            noisy_target=w_star + noise,
        )

    if variant == "completion":
        num_obs = max(1, int(round(completion_multiplier * r_star * (m + n))))
        num_obs = min(num_obs, m * n)
        flat_idx = torch.randperm(m * n, generator=g, device=device)[:num_obs]
        rows = torch.div(flat_idx, n, rounding_mode="floor")
        cols = flat_idx % n
        values = w_star[rows, cols]
        if sigma > 0.0:
            values = values + torch.randn(values.shape, generator=g, device=device, dtype=dtype) * sigma
        return SyntheticProblem(
            variant=variant,
            m=m,
            n=n,
            r_star=r_star,
            kappa=kappa,
            snr_db=snr_db,
            w_star=w_star,
            rows=rows,
            cols=cols,
            values=values,
        )

    if variant == "sensing":
        num_meas = max(1, int(round(sensing_multiplier * r_star * (m + n))))
        ops = torch.randn((num_meas, m, n), generator=g, device=device, dtype=dtype) / math.sqrt(m * n)
        targets = torch.einsum("pmn,mn->p", ops, w_star)
        if sigma > 0.0:
            targets = targets + torch.randn(targets.shape, generator=g, device=device, dtype=dtype) * sigma
        return SyntheticProblem(
            variant=variant,
            m=m,
            n=n,
            r_star=r_star,
            kappa=kappa,
            snr_db=snr_db,
            w_star=w_star,
            sensing_targets=targets,
            sensing_seed=seed + 10_000,
            sensing_chunk_size=sensing_chunk_size,
            sensing_num_meas=num_meas,
        )

    raise ValueError(f"Unknown synthetic variant: {variant}")


def init_factors(
    *,
    m: int,
    n: int,
    rank: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> LowRankState:
    g = torch.Generator(device=device).manual_seed(seed)
    scale = 1.0 / math.sqrt(rank)
    b = torch.randn((m, rank), generator=g, device=device, dtype=dtype) * scale
    a = torch.randn((rank, n), generator=g, device=device, dtype=dtype) * scale
    return LowRankState(b=b, a=a)


def scale_factor_gauge(state: LowRankState, alpha: float) -> LowRankState:
    alpha_t = state.b.new_tensor(float(alpha))
    return LowRankState(b=state.b * alpha_t, a=state.a / alpha_t)


def relative_recovery_error(state: LowRankState, w_star: torch.Tensor) -> float:
    diff = matrix_from_state(state) - w_star
    return float((torch.linalg.norm(diff, ord="fro") / torch.linalg.norm(w_star, ord="fro").clamp_min(1e-30)).item())


def loss_and_factor_grads(
    problem: SyntheticProblem,
    state: LowRankState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = matrix_from_state(state)

    if problem.variant == "full":
        assert problem.noisy_target is not None
        residual = w - problem.noisy_target
        loss = torch.mean(residual * residual)
        grad_w = (2.0 / float(problem.m * problem.n)) * residual
        return loss, grad_w @ state.a.transpose(-1, -2), state.b.transpose(-1, -2) @ grad_w

    if problem.variant == "completion":
        assert problem.rows is not None and problem.cols is not None and problem.values is not None
        b_rows = state.b.index_select(0, problem.rows)
        a_cols = state.a.transpose(-1, -2).index_select(0, problem.cols)
        residual = torch.sum(b_rows * a_cols, dim=1) - problem.values
        loss = torch.mean(residual * residual)
        scale = 2.0 / float(problem.values.numel())
        grad_b = torch.zeros_like(state.b)
        grad_b.index_add_(0, problem.rows, scale * residual.unsqueeze(1) * a_cols)
        grad_a_t = torch.zeros((problem.n, state.a.shape[0]), dtype=state.a.dtype, device=state.a.device)
        grad_a_t.index_add_(0, problem.cols, scale * residual.unsqueeze(1) * b_rows)
        return loss, grad_b, grad_a_t.transpose(-1, -2)

    if problem.variant == "sensing":
        assert problem.sensing_targets is not None
        assert problem.sensing_seed is not None
        assert problem.sensing_num_meas is not None
        g = torch.Generator(device=w.device).manual_seed(problem.sensing_seed)
        chunk_size = max(1, int(problem.sensing_chunk_size))
        preds_chunks: list[torch.Tensor] = []
        grad_w = torch.zeros_like(w)
        offset = 0
        while offset < problem.sensing_num_meas:
            chunk = min(chunk_size, problem.sensing_num_meas - offset)
            ops = torch.randn((chunk, problem.m, problem.n), generator=g, device=w.device, dtype=w.dtype)
            ops = ops / math.sqrt(problem.m * problem.n)
            targets = problem.sensing_targets[offset : offset + chunk]
            preds = torch.einsum("pmn,mn->p", ops, w)
            residual = preds - targets
            preds_chunks.append(preds)
            grad_w = grad_w + torch.einsum("p,pmn->mn", residual, ops)
            offset += chunk
        preds_all = torch.cat(preds_chunks)
        residual_all = preds_all - problem.sensing_targets
        loss = 0.5 * torch.mean(residual_all * residual_all)
        grad_w = grad_w / float(problem.sensing_targets.numel())
        return loss, grad_w @ state.a.transpose(-1, -2), state.b.transpose(-1, -2) @ grad_w

    raise ValueError(f"Unknown synthetic variant: {problem.variant}")
