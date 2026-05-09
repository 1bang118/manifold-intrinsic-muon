from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.utils import DEFAULT_DTYPE, ensure_dir, format_lr, hardware_info, matrix_power_symmetric, ortho, save_json, setup_seed, write_csv


torch.set_default_dtype(DEFAULT_DTYPE)


DEFAULT_LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
SMOKE_LR_GRID = [1e-2, 1e-1]
DEFAULT_RANKS = [5, 10, 20]
SMOKE_RANKS = [4]
DEFAULT_METHODS = [
    "riemannian_gd",
    "scaled_muon",
    "scaled_numuon",
    "euclidean_gd",
    "spectron_muon",
    "euclidean_muon",
    "euclidean_numuon",
]


@dataclass(frozen=True)
class RatingsData:
    n_users: int
    n_movies: int
    users: np.ndarray
    movies: np.ndarray
    ratings: np.ndarray
    source: str


@dataclass(frozen=True)
class SplitData:
    train_users: torch.Tensor
    train_movies: torch.Tensor
    train_ratings: torch.Tensor
    val_users: torch.Tensor
    val_movies: torch.Tensor
    val_ratings: torch.Tensor
    test_users: torch.Tensor
    test_movies: torch.Tensor
    test_ratings: torch.Tensor
    global_mean: float
    user_bias: torch.Tensor
    movie_bias: torch.Tensor
    split_indices: dict[str, np.ndarray]


@dataclass(frozen=True)
class FactorState:
    b: torch.Tensor
    a: torch.Tensor


def parse_ratings_dat(path: Path) -> RatingsData:
    users: list[int] = []
    movies: list[int] = []
    ratings: list[float] = []
    max_user = 0
    max_movie = 0
    with path.open("r", encoding="latin-1") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            user_s, movie_s, rating_s, *_ = line.split("::")
            user = int(user_s) - 1
            movie = int(movie_s) - 1
            rating = float(rating_s)
            users.append(user)
            movies.append(movie)
            ratings.append(rating)
            max_user = max(max_user, user + 1)
            max_movie = max(max_movie, movie + 1)
    return RatingsData(
        n_users=max_user,
        n_movies=max_movie,
        users=np.asarray(users, dtype=np.int64),
        movies=np.asarray(movies, dtype=np.int64),
        ratings=np.asarray(ratings, dtype=np.float64),
        source=str(path),
    )


def make_synthetic_ratings(
    *,
    seed: int,
    n_users: int = 120,
    n_movies: int = 90,
    true_rank: int = 5,
    observed_fraction: float = 0.25,
    noise_std: float = 0.1,
) -> RatingsData:
    rng = np.random.default_rng(seed)
    b_true = rng.normal(scale=1.0 / math.sqrt(true_rank), size=(n_users, true_rank))
    a_true = rng.normal(scale=1.0 / math.sqrt(true_rank), size=(true_rank, n_movies))
    full = b_true @ a_true
    full = 3.0 + full / np.std(full)
    full = np.clip(full + rng.normal(scale=noise_std, size=full.shape), 1.0, 5.0)
    mask = rng.random(full.shape) < observed_fraction
    users, movies = np.nonzero(mask)
    ratings = full[users, movies]
    return RatingsData(
        n_users=n_users,
        n_movies=n_movies,
        users=users.astype(np.int64),
        movies=movies.astype(np.int64),
        ratings=ratings.astype(np.float64),
        source="synthetic",
    )


def load_ratings(args: argparse.Namespace) -> RatingsData:
    if args.synthetic:
        return make_synthetic_ratings(seed=args.seed)
    path = Path(args.data_path)
    if path.is_dir():
        path = path / "ratings.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Point --data-path to MovieLens-1M/ratings.dat or run with --synthetic."
        )
    return parse_ratings_dat(path)


def estimate_biases(
    n_users: int,
    n_movies: int,
    users: np.ndarray,
    movies: np.ndarray,
    ratings: np.ndarray,
    *,
    reg: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    global_mean = float(ratings.mean())
    user_sum = np.zeros(n_users, dtype=np.float64)
    user_count = np.zeros(n_users, dtype=np.float64)
    residual = ratings - global_mean
    np.add.at(user_sum, users, residual)
    np.add.at(user_count, users, 1.0)
    user_bias = user_sum / np.maximum(user_count + reg, 1.0)

    movie_sum = np.zeros(n_movies, dtype=np.float64)
    movie_count = np.zeros(n_movies, dtype=np.float64)
    residual = ratings - global_mean - user_bias[users]
    np.add.at(movie_sum, movies, residual)
    np.add.at(movie_count, movies, 1.0)
    movie_bias = movie_sum / np.maximum(movie_count + reg, 1.0)
    return global_mean, user_bias, movie_bias


def split_observations(
    data: RatingsData,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
    device: torch.device,
    *,
    bias_reg: float,
) -> SplitData:
    rng = np.random.default_rng(seed)
    n_obs = len(data.ratings)
    perm = rng.permutation(n_obs)
    n_test = max(1, int(round(test_fraction * n_obs)))
    n_test = min(n_test, n_obs - 2)
    remaining = n_obs - n_test
    n_val = max(1, int(round(validation_fraction * remaining)))
    n_val = min(n_val, remaining - 1)
    test_idx = np.sort(perm[:n_test])
    val_idx = np.sort(perm[n_test : n_test + n_val])
    train_idx = np.sort(perm[n_test + n_val :])

    global_mean, user_bias_np, movie_bias_np = estimate_biases(
        data.n_users,
        data.n_movies,
        data.users[train_idx],
        data.movies[train_idx],
        data.ratings[train_idx],
        reg=bias_reg,
    )

    def residual_targets(indices: np.ndarray) -> np.ndarray:
        return (
            data.ratings[indices]
            - global_mean
            - user_bias_np[data.users[indices]]
            - movie_bias_np[data.movies[indices]]
        )

    def tensor(arr: np.ndarray, dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
        return torch.as_tensor(arr, device=device, dtype=dtype)

    return SplitData(
        train_users=tensor(data.users[train_idx], dtype=torch.long),
        train_movies=tensor(data.movies[train_idx], dtype=torch.long),
        train_ratings=tensor(residual_targets(train_idx)),
        val_users=tensor(data.users[val_idx], dtype=torch.long),
        val_movies=tensor(data.movies[val_idx], dtype=torch.long),
        val_ratings=tensor(residual_targets(val_idx)),
        test_users=tensor(data.users[test_idx], dtype=torch.long),
        test_movies=tensor(data.movies[test_idx], dtype=torch.long),
        test_ratings=tensor(residual_targets(test_idx)),
        global_mean=global_mean,
        user_bias=tensor(user_bias_np),
        movie_bias=tensor(movie_bias_np),
        split_indices={"train": train_idx, "val": val_idx, "test": test_idx},
    )


def init_factors(n_users: int, n_movies: int, rank: int, seed: int, device: torch.device) -> FactorState:
    g = torch.Generator(device=device).manual_seed(seed)
    scale = 1.0 / math.sqrt(rank)
    b = torch.randn((n_users, rank), generator=g, device=device, dtype=DEFAULT_DTYPE) * scale
    a = torch.randn((rank, n_movies), generator=g, device=device, dtype=DEFAULT_DTYPE) * scale
    return FactorState(b=b, a=a)


def init_factors_from_observed_svd(
    split: SplitData,
    n_users: int,
    n_movies: int,
    rank: int,
    seed: int,
    device: torch.device,
    *,
    oversampling: int,
    niter: int,
) -> FactorState:
    observed = torch.zeros((n_users, n_movies), dtype=DEFAULT_DTYPE, device=device)
    observed[split.train_users, split.train_movies] = split.train_ratings
    q = min(min(n_users, n_movies), max(rank + oversampling, rank))
    torch.manual_seed(seed + 91_337)
    u, s, v = torch.svd_lowrank(observed, q=q, niter=niter)
    u_r = u[:, :rank]
    s_r = s[:rank].clamp_min(1e-16)
    v_r = v[:, :rank]
    sqrt_s = torch.sqrt(s_r)
    b = u_r * sqrt_s.unsqueeze(0)
    a = sqrt_s.unsqueeze(1) * v_r.transpose(0, 1)
    return FactorState(b=b.contiguous(), a=a.contiguous())


def make_initial_state(
    split: SplitData,
    n_users: int,
    n_movies: int,
    rank: int,
    seed: int,
    device: torch.device,
    *,
    init_mode: str,
    svd_oversampling: int,
    svd_niter: int,
) -> FactorState:
    if init_mode == "random":
        return init_factors(n_users, n_movies, rank, seed, device)
    if init_mode == "observed-svd":
        return init_factors_from_observed_svd(
            split,
            n_users,
            n_movies,
            rank,
            seed,
            device,
            oversampling=svd_oversampling,
            niter=svd_niter,
        )
    raise ValueError(f"Unknown init mode: {init_mode}")


def save_shared_artifacts(shared_dir: Path, rank: int, seed: int, split: SplitData, init_state: FactorState) -> None:
    ensure_dir(shared_dir)
    torch.save(
        {
            "train_indices": split.split_indices["train"],
            "val_indices": split.split_indices["val"],
            "test_indices": split.split_indices["test"],
            "global_mean": split.global_mean,
            "user_bias": split.user_bias.detach().cpu(),
            "movie_bias": split.movie_bias.detach().cpu(),
        },
        shared_dir / f"split_seed{seed}_rank{rank}.pt",
    )
    torch.save(
        {
            "b": init_state.b.detach().cpu(),
            "a": init_state.a.detach().cpu(),
        },
        shared_dir / f"init_seed{seed}_rank{rank}.pt",
    )


def safe_invsqrt_spd(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return matrix_power_symmetric(x + eps * eye, -0.5, eps=eps)


def factor_grams(state: FactorState, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    s_b = state.b.transpose(-1, -2) @ state.b
    s_a = state.a @ state.a.transpose(-1, -2)
    eye = torch.eye(s_b.shape[-1], dtype=state.b.dtype, device=state.b.device)
    return s_b + eps * eye, s_a + eps * eye


def predict_entries(state: FactorState, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
    b_rows = state.b.index_select(0, users)
    a_cols = state.a.transpose(0, 1).index_select(0, movies)
    return torch.sum(b_rows * a_cols, dim=1)


def mse_loss(state: FactorState, users: torch.Tensor, movies: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
    preds = predict_entries(state, users, movies)
    return torch.mean((preds - ratings) ** 2)


def objective_loss(
    state: FactorState,
    users: torch.Tensor,
    movies: torch.Tensor,
    ratings: torch.Tensor,
    *,
    l2_reg: float = 0.0,
    l2_target: str = "product",
) -> torch.Tensor:
    loss = mse_loss(state, users, movies, ratings)
    if l2_reg <= 0.0:
        return loss
    if l2_target == "product":
        product = state.b @ state.a
        penalty = torch.mean(product * product)
    elif l2_target == "factors":
        penalty = (torch.mean(state.b * state.b) + torch.mean(state.a * state.a)) / 2.0
    else:
        raise ValueError(f"Unknown L2 target: {l2_target}")
    return loss + 0.5 * float(l2_reg) * penalty


def rmse(state: FactorState, users: torch.Tensor, movies: torch.Tensor, ratings: torch.Tensor) -> float:
    with torch.no_grad():
        preds = predict_entries(state, users, movies)
        return float(torch.sqrt(torch.mean((preds - ratings) ** 2)).item())


def condition_number(x: torch.Tensor, eps: float = 1e-12) -> float:
    s = torch.linalg.svdvals(x)
    if s.numel() == 0:
        return 0.0
    return float((s.max() / s.min().clamp_min(eps)).item())


def factor_norm_sq(state: FactorState) -> float:
    return float((state.b.pow(2).sum() + state.a.pow(2).sum()).item())


def pair_fro_norm(x_b: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(x_b * x_b) + torch.sum(x_a * x_a))


def pair_dual_norm(h_b: torch.Tensor, h_a: torch.Tensor, norm_type: str) -> float:
    if norm_type == "frobenius":
        return float(pair_fro_norm(h_b, h_a).item())
    if norm_type == "spectral":
        return float(torch.linalg.norm(h_b, ord="nuc").item() + torch.linalg.norm(h_a, ord="nuc").item())
    if norm_type == "nuclear":
        return float(max(torch.linalg.norm(h_b, ord=2).item(), torch.linalg.norm(h_a, ord=2).item()))
    raise ValueError(norm_type)


def whitened_gradients(state: FactorState, egrad_b: torch.Tensor, egrad_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_b, s_a = factor_grams(state)
    invsqrt_b = safe_invsqrt_spd(s_b)
    invsqrt_a = safe_invsqrt_spd(s_a)
    h_b = egrad_b @ invsqrt_a
    h_a = invsqrt_b @ egrad_a
    return h_b, h_a, invsqrt_b, invsqrt_a


def riemannian_gd_step(state: FactorState, egrad_b: torch.Tensor, egrad_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h_b, h_a, invsqrt_b, invsqrt_a = whitened_gradients(state, egrad_b, egrad_a)
    denom = pair_fro_norm(h_b, h_a)
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    z_b = h_b / denom
    z_a = h_a / denom
    return z_b @ invsqrt_a, invsqrt_b @ z_a


def scaled_muon_step(state: FactorState, egrad_b: torch.Tensor, egrad_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h_b, h_a, invsqrt_b, invsqrt_a = whitened_gradients(state, egrad_b, egrad_a)
    if float(pair_fro_norm(h_b, h_a).item()) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    z_b = ortho(h_b)
    z_a = ortho(h_a.transpose(-1, -2)).transpose(-1, -2)
    return z_b @ invsqrt_a, invsqrt_b @ z_a


def _rank1(x: torch.Tensor) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    if s.numel() == 0 or float(s[0].item()) <= 1e-14:
        return torch.zeros_like(x)
    return u[:, :1] @ vh[:1, :]


def scaled_numuon_step(state: FactorState, egrad_b: torch.Tensor, egrad_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h_b, h_a, invsqrt_b, invsqrt_a = whitened_gradients(state, egrad_b, egrad_a)
    if max(float(torch.linalg.norm(h_b, ord=2).item()), float(torch.linalg.norm(h_a, ord=2).item())) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    z_b = _rank1(h_b)
    z_a = _rank1(h_a)
    return z_b @ invsqrt_a, invsqrt_b @ z_a


def euclidean_gd_step(state: FactorState, egrad_b: torch.Tensor, egrad_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denom = pair_fro_norm(egrad_b, egrad_a)
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    return egrad_b / denom, egrad_a / denom


def euclidean_muon_step(state: FactorState, egrad_b: torch.Tensor, egrad_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if float(pair_fro_norm(egrad_b, egrad_a).item()) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    return ortho(egrad_b), ortho(egrad_a.transpose(-1, -2)).transpose(-1, -2)


def spectron_muon_step(state: FactorState, egrad_b: torch.Tensor, egrad_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if float(pair_fro_norm(egrad_b, egrad_a).item()) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    denom = float(torch.linalg.norm(state.b, ord=2).item() + torch.linalg.norm(state.a, ord=2).item() + 1.0)
    if denom <= 0.0 or not math.isfinite(denom):
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    return ortho(egrad_b) / denom, ortho(egrad_a.transpose(-1, -2)).transpose(-1, -2) / denom


def euclidean_numuon_step(state: FactorState, egrad_b: torch.Tensor, egrad_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if max(float(torch.linalg.norm(egrad_b, ord=2).item()), float(torch.linalg.norm(egrad_a, ord=2).item())) <= 1e-14:
        return torch.zeros_like(state.b), torch.zeros_like(state.a)
    return _rank1(egrad_b), _rank1(egrad_a)


METHODS: dict[str, dict[str, Any]] = {
    "riemannian_gd": {"step_fn": riemannian_gd_step, "norm": "frobenius", "geometry": "coupled"},
    "scaled_muon": {"step_fn": scaled_muon_step, "norm": "spectral", "geometry": "coupled"},
    "scaled_numuon": {"step_fn": scaled_numuon_step, "norm": "nuclear", "geometry": "coupled"},
    "euclidean_gd": {"step_fn": euclidean_gd_step, "norm": "frobenius", "geometry": "euclidean"},
    "spectron_muon": {"step_fn": spectron_muon_step, "norm": "spectral", "geometry": "euclidean"},
    "euclidean_muon": {"step_fn": euclidean_muon_step, "norm": "spectral", "geometry": "euclidean"},
    "euclidean_numuon": {"step_fn": euclidean_numuon_step, "norm": "nuclear", "geometry": "euclidean"},
}


def retract_factors(state: FactorState, xi_b: torch.Tensor, xi_a: torch.Tensor, lr: float, *, min_sigma: float = 1e-8, max_backtracks: int = 12) -> FactorState:
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
        cand = FactorState(b=b_new.detach(), a=a_new.detach())
        last = cand
        if smin_b > min_sigma and smin_a > min_sigma:
            return cand
        step *= 0.5
    return last


def method_metrics(state: FactorState, egrad_b: torch.Tensor, egrad_a: torch.Tensor, xi_b: torch.Tensor, xi_a: torch.Tensor, norm_type: str, geometry: str) -> dict[str, float]:
    if geometry == "coupled":
        h_b, h_a, invsqrt_b, invsqrt_a = whitened_gradients(state, egrad_b, egrad_a)
        z_b = xi_b @ matrix_power_symmetric(factor_grams(state)[1], 0.5)
        z_a = matrix_power_symmetric(factor_grams(state)[0], 0.5) @ xi_a
        rgrad_b = egrad_b @ matrix_power_symmetric(factor_grams(state)[1], -1.0)
        rgrad_a = matrix_power_symmetric(factor_grams(state)[0], -1.0) @ egrad_a
        rgrad_norm = float(pair_fro_norm(h_b, h_a).item())
    else:
        h_b, h_a = egrad_b, egrad_a
        z_b, z_a = xi_b, xi_a
        rgrad_b, rgrad_a = egrad_b, egrad_a
        rgrad_norm = float(pair_fro_norm(rgrad_b, rgrad_a).item())
    return {
        "rgrad_norm": rgrad_norm,
        "dual_norm_H": pair_dual_norm(h_b, h_a, norm_type),
        "Z_norm_sq": float((torch.sum(z_b * z_b) + torch.sum(z_a * z_a)).item()),
        "kappa_B": condition_number(state.b),
        "kappa_A": condition_number(state.a),
        "factor_norm_sq": factor_norm_sq(state),
    }


def convergence_row(
    epoch: int,
    train_loss: float,
    test_rmse: float,
    wall_time_sec: float,
    lr_t: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    row = {
        "epoch": epoch,
        "lr_t": lr_t,
        "train_loss": train_loss,
        "test_rmse": test_rmse,
        "wall_time_sec": wall_time_sec,
    }
    row.update(metrics)
    return row


def run_one_training(
    split: SplitData,
    init_state: FactorState,
    method: str,
    lr: float,
    max_epochs: int,
    device: torch.device,
    schedule: str = "constant",
    l2_reg: float = 0.0,
    l2_target: str = "product",
) -> tuple[FactorState, list[dict[str, Any]], dict[str, Any]]:
    cfg = METHODS[method]
    state = FactorState(b=init_state.b.clone(), a=init_state.a.clone())
    logs: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        b_var = state.b.detach().clone().requires_grad_(True)
        a_var = state.a.detach().clone().requires_grad_(True)
        loss = objective_loss(
            FactorState(b=b_var, a=a_var),
            split.train_users,
            split.train_movies,
            split.train_ratings,
            l2_reg=l2_reg,
            l2_target=l2_target,
        )
        loss.backward()
        egrad_b = b_var.grad.detach()
        egrad_a = a_var.grad.detach()
        xi_b, xi_a = cfg["step_fn"](state, egrad_b, egrad_a)
        metrics = method_metrics(state, egrad_b, egrad_a, xi_b, xi_a, cfg["norm"], cfg["geometry"])
        lr_t = float(lr)
        if schedule == "sqrt":
            lr_t = lr_t / math.sqrt(epoch)
        elif schedule == "linear":
            lr_t = lr_t / float(epoch)
        elif schedule != "constant":
            raise ValueError(f"Unknown learning-rate schedule: {schedule}")
        state = retract_factors(state, xi_b, xi_a, lr_t)
        elapsed = time.perf_counter() - t0
        logs.append(
            convergence_row(
                epoch=epoch,
                lr_t=lr_t,
                train_loss=float(loss.item()),
                test_rmse=rmse(state, split.test_users, split.test_movies, split.test_ratings),
                wall_time_sec=elapsed,
                metrics=metrics,
            )
        )
    final = logs[-1]
    summary = {
        "method": method,
        "lr": lr,
        "schedule": schedule,
        "l2_reg": float(l2_reg),
        "l2_target": l2_target,
        "final_train_loss": final["train_loss"],
        "final_val_rmse": rmse(state, split.val_users, split.val_movies, split.val_ratings),
        "final_test_rmse": final["test_rmse"],
        "final_rgrad_norm": final["rgrad_norm"],
        "final_dual_norm_H": final["dual_norm_H"],
        "final_Z_norm_sq": final["Z_norm_sq"],
        "final_kappa_B": final["kappa_B"],
        "final_kappa_A": final["kappa_A"],
        "final_factor_norm_sq": final["factor_norm_sq"],
        "final_wall_time_sec": final["wall_time_sec"],
    }
    return state, logs, summary


def save_run_outputs(run_dir: Path, config: dict[str, Any], init_path: Path, logs: list[dict[str, Any]], final_state: FactorState, summary: dict[str, Any]) -> None:
    ensure_dir(run_dir)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "init_checkpoint.json", {"path": str(init_path)})
    write_csv(run_dir / "metrics.csv", logs)
    torch.save({"b": final_state.b.detach().cpu(), "a": final_state.a.detach().cpu()}, run_dir / "final_model.pt")
    save_json(run_dir / "summary.json", summary)


def rank_sweep_summary(results_dir: Path, rank: int, methods: list[str], seeds: list[int], lr_grid: list[float]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    best: dict[str, float] = {}
    for method in methods:
        candidates = []
        for lr in lr_grid:
            values = []
            val_values = []
            for seed in seeds:
                p = results_dir / f"seed{seed}" / f"rank{rank}" / method / f"lr_{format_lr(lr)}" / "summary.json"
                if p.exists():
                    import json
                    d = json.loads(p.read_text())
                    val_values.append(float(d["final_val_rmse"]))
                    values.append(float(d["final_test_rmse"]))
            if values and val_values:
                row = {
                    "rank": rank,
                    "method": method,
                    "lr": float(lr),
                    "n_seeds": len(values),
                    "mean_final_val_rmse": float(np.mean(val_values)),
                    "std_final_val_rmse": float(np.std(val_values, ddof=1)) if len(val_values) > 1 else 0.0,
                    "mean_final_test_rmse": float(np.mean(values)),
                    "std_final_test_rmse": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
                rows.append(row)
                candidates.append(row)
        if candidates:
            chosen = min(candidates, key=lambda row: (row["mean_final_val_rmse"], row["mean_final_test_rmse"], -row["n_seeds"], row["lr"]))
            best[method] = float(chosen["lr"])
    return rows, best


def aggregate_best_runs(results_dir: Path, rank: int, best_lrs: dict[str, float], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for method, lr in best_lrs.items():
        values = {"final_val_rmse": [], "final_test_rmse": [], "final_train_loss": [], "final_wall_time_sec": []}
        for seed in seeds:
            p = results_dir / f"seed{seed}" / f"rank{rank}" / method / f"lr_{format_lr(lr)}" / "summary.json"
            if not p.exists():
                continue
            import json
            d = json.loads(p.read_text())
            for key in values:
                values[key].append(float(d[key]))
        rows.append(
            {
                "rank": rank,
                "method": method,
                "best_lr": lr,
                "n_seeds": len(values["final_test_rmse"]),
                "mean_final_val_rmse": float(np.mean(values["final_val_rmse"])) if values["final_val_rmse"] else None,
                "mean_final_test_rmse": float(np.mean(values["final_test_rmse"])) if values["final_test_rmse"] else None,
                "std_final_test_rmse": float(np.std(values["final_test_rmse"], ddof=1)) if len(values["final_test_rmse"]) > 1 else 0.0 if values["final_test_rmse"] else None,
                "mean_final_train_loss": float(np.mean(values["final_train_loss"])) if values["final_train_loss"] else None,
                "mean_final_wall_time_sec": float(np.mean(values["final_wall_time_sec"])) if values["final_wall_time_sec"] else None,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-rank MovieLens matrix completion with intrinsic and Euclidean Muon-style baselines.")
    parser.add_argument("--data-path", default="data/movielens/ml-1m/ratings.dat")
    parser.add_argument("--results-dir", default="results/movielens_fixed_rank")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--ranks", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None, choices=list(METHODS.keys()))
    parser.add_argument("--lr-grid", type=float, nargs="+", default=None)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--schedule", choices=["constant", "sqrt", "linear"], default="constant")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--bias-reg", type=float, default=10.0)
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--l2-target", choices=["product", "factors"], default="product")
    parser.add_argument("--init-mode", choices=["random", "observed-svd"], default="random")
    parser.add_argument("--svd-oversampling", type=int, default=8)
    parser.add_argument("--svd-niter", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.seeds is None:
        args.seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    if args.ranks is None:
        args.ranks = SMOKE_RANKS if args.smoke else DEFAULT_RANKS
    if args.methods is None:
        args.methods = DEFAULT_METHODS
    if args.lr_grid is None:
        args.lr_grid = SMOKE_LR_GRID if args.smoke else DEFAULT_LR_GRID
    if args.smoke:
        args.synthetic = True
        args.max_epochs = min(args.max_epochs, 5)
    args.seeds = [int(v) for v in args.seeds]
    args.ranks = [int(v) for v in args.ranks]
    return args


def main() -> None:
    args = normalize_args(parse_args())
    setup_seed(args.seed)
    device = torch.device(args.device)
    results_dir = ensure_dir(args.results_dir)
    data = load_ratings(args)

    save_json(
        results_dir / "run_config.json",
        {
            "data_source": data.source,
            "n_users": data.n_users,
            "n_movies": data.n_movies,
            "n_observations": int(len(data.ratings)),
            "methods": args.methods,
            "ranks": args.ranks,
            "seeds": args.seeds,
            "lr_grid": args.lr_grid,
            "max_epochs": args.max_epochs,
            "schedule": args.schedule,
            "test_fraction": args.test_fraction,
            "validation_fraction": args.validation_fraction,
            "bias_reg": args.bias_reg,
            "l2_reg": args.l2_reg,
            "l2_target": args.l2_target,
            "init_mode": args.init_mode,
            "svd_oversampling": args.svd_oversampling,
            "svd_niter": args.svd_niter,
            "synthetic": args.synthetic,
            "smoke": args.smoke,
            "device": str(device),
            "hardware": hardware_info(),
        },
    )

    shared_dir = ensure_dir(results_dir / "shared")
    for seed in args.seeds:
        split = split_observations(
            data,
            seed,
            args.test_fraction,
            args.validation_fraction,
            device,
            bias_reg=args.bias_reg,
        )
        for rank in args.ranks:
            init_state = make_initial_state(
                split,
                data.n_users,
                data.n_movies,
                rank,
                seed,
                device,
                init_mode=args.init_mode,
                svd_oversampling=args.svd_oversampling,
                svd_niter=args.svd_niter,
            )
            save_shared_artifacts(shared_dir, rank, seed, split, init_state)

            for method in args.methods:
                for lr in args.lr_grid:
                    run_dir = ensure_dir(results_dir / f"seed{seed}" / f"rank{rank}" / method / f"lr_{format_lr(lr)}")
                    init_path = shared_dir / f"init_seed{seed}_rank{rank}.pt"
                    config = {
                        "seed": seed,
                        "rank": rank,
                        "method": method,
                        "lr": lr,
                        "schedule": args.schedule,
                        "max_epochs": args.max_epochs,
                        "test_fraction": args.test_fraction,
                        "data_source": data.source,
                        "n_users": data.n_users,
                        "n_movies": data.n_movies,
                        "n_train_observations": int(split.train_ratings.numel()),
                        "n_val_observations": int(split.val_ratings.numel()),
                        "n_test_observations": int(split.test_ratings.numel()),
                        "global_mean": split.global_mean,
                        "l2_reg": args.l2_reg,
                        "l2_target": args.l2_target,
                        "init_mode": args.init_mode,
                        "svd_oversampling": args.svd_oversampling,
                        "svd_niter": args.svd_niter,
                        "norm": METHODS[method]["norm"],
                        "geometry": METHODS[method]["geometry"],
                    }
                    final_state, logs, summary = run_one_training(
                        split,
                        init_state,
                        method,
                        float(lr),
                        args.max_epochs,
                        device,
                        schedule=args.schedule,
                        l2_reg=args.l2_reg,
                        l2_target=args.l2_target,
                    )
                    save_run_outputs(run_dir, config, init_path, logs, final_state, summary)

    sweep_rows_all = []
    best_lr_payload: dict[str, dict[str, float]] = {}
    aggregate_rows = []
    for rank in args.ranks:
        sweep_rows, best = rank_sweep_summary(results_dir, rank, args.methods, args.seeds, args.lr_grid)
        sweep_rows_all.extend(sweep_rows)
        best_lr_payload[f"rank{rank}"] = best
        aggregate_rows.extend(aggregate_best_runs(results_dir, rank, best, args.seeds))

    write_csv(results_dir / "lr_sweep_summary.csv", sweep_rows_all)
    save_json(results_dir / "global_best_lr.json", best_lr_payload)
    write_csv(results_dir / "mean_test_rmse_table.csv", aggregate_rows)


if __name__ == "__main__":
    main()
