from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manifolds.lowrank import METHODS, LowRankState, factor_grams, method_metrics, retract_factors
from src.utils import DEFAULT_DTYPE, ensure_dir, format_lr, hardware_info, save_json, setup_seed, write_csv


torch.set_default_dtype(DEFAULT_DTYPE)


DEFAULT_METHODS = [
    "riemannian_gd",
    "scaled_muon",
    "scaled_numuon",
    "euclidean_gd",
    "euclidean_muon",
    "euclidean_numuon",
]

METHOD_LABELS = {
    "riemannian_gd": "RGD",
    "scaled_muon": "iMuon",
    "scaled_numuon": "iMuon-Nu",
    "euclidean_gd": "EGD",
    "euclidean_muon": "Muon",
    "euclidean_numuon": "NuMuon",
}

PAIR_SPECS = [
    ("Frobenius", "riemannian_gd", "euclidean_gd"),
    ("Spectral", "scaled_muon", "euclidean_muon"),
    ("Nuclear", "scaled_numuon", "euclidean_numuon"),
]


@dataclass(frozen=True)
class LowRankCompletionProblem:
    m: int
    n: int
    r_star: int
    kappa: float
    u_star: torch.Tensor
    sigmas: torch.Tensor
    v_star: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor
    snr_db: float


def parse_size_spec(spec: str) -> tuple[int, int]:
    left, right = spec.lower().split("x", maxsplit=1)
    return int(left), int(right)


def parse_snr(token: str) -> float:
    token = str(token).lower()
    if token in {"inf", "infty", "infinity"}:
        return math.inf
    return float(token)


def snr_label(snr_db: float) -> str:
    return "inf" if math.isinf(float(snr_db)) else format_lr(float(snr_db))


def kappa_label(kappa: float) -> str:
    return str(int(kappa)) if float(kappa).is_integer() else format_lr(float(kappa))


def generate_ground_truth_factors(
    *,
    m: int,
    n: int,
    r_star: int,
    kappa: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    u_raw = torch.randn((m, r_star), generator=g, device=device, dtype=dtype)
    v_raw = torch.randn((n, r_star), generator=g, device=device, dtype=dtype)
    u_star, _ = torch.linalg.qr(u_raw)
    v_star, _ = torch.linalg.qr(v_raw)
    if r_star == 1:
        sigmas = torch.ones((1,), device=device, dtype=dtype)
    else:
        powers = torch.arange(r_star, device=device, dtype=dtype) / float(r_star - 1)
        sigmas = torch.tensor(float(kappa), device=device, dtype=dtype).pow(-powers)
    sigmas = sigmas / torch.linalg.norm(sigmas).clamp_min(1e-30)
    return u_star, sigmas, v_star


def sample_values_from_lowrank(
    u_star: torch.Tensor,
    sigmas: torch.Tensor,
    v_star: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    u_rows = u_star.index_select(0, rows)
    v_cols = v_star.index_select(0, cols)
    return torch.sum(u_rows * sigmas.unsqueeze(0) * v_cols, dim=1)


def noise_std_from_snr_db(snr_db: float, num_values: int) -> float:
    if math.isinf(float(snr_db)):
        return 0.0
    target_rms = 10.0 ** (-float(snr_db) / 20.0)
    return target_rms / math.sqrt(max(1, num_values))


def build_completion_problem(
    *,
    m: int,
    n: int,
    r_star: int,
    kappa: float,
    seed: int,
    completion_multiplier: float,
    snr_db: float,
    device: torch.device,
) -> LowRankCompletionProblem:
    u_star, sigmas, v_star = generate_ground_truth_factors(
        m=m,
        n=n,
        r_star=r_star,
        kappa=kappa,
        seed=seed,
        device=device,
    )
    num_obs = max(1, int(round(float(completion_multiplier) * r_star * (m + n))))
    g = torch.Generator(device=device).manual_seed(seed + 10_000)
    rows = torch.randint(0, m, (num_obs,), generator=g, device=device)
    cols = torch.randint(0, n, (num_obs,), generator=g, device=device)
    values = sample_values_from_lowrank(u_star, sigmas, v_star, rows, cols)
    sigma = noise_std_from_snr_db(snr_db, num_obs)
    if sigma > 0.0:
        values = values + torch.randn(values.shape, generator=g, device=device, dtype=values.dtype) * sigma
    return LowRankCompletionProblem(
        m=m,
        n=n,
        r_star=r_star,
        kappa=float(kappa),
        u_star=u_star,
        sigmas=sigmas,
        v_star=v_star,
        rows=rows,
        cols=cols,
        values=values,
        snr_db=float(snr_db),
    )


def init_factors(
    *,
    m: int,
    n: int,
    rank: int,
    seed: int,
    device: torch.device,
    product_norm: float | None,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> LowRankState:
    g = torch.Generator(device=device).manual_seed(seed)
    if product_norm is None:
        scale = 1.0 / math.sqrt(rank)
    else:
        # For iid factors with entry scale s, E||BA||_F^2 = m n rank s^4.
        scale = math.sqrt(float(product_norm)) / ((float(m) * float(n) * float(rank)) ** 0.25)
    b = torch.randn((m, rank), generator=g, device=device, dtype=dtype) * scale
    a = torch.randn((rank, n), generator=g, device=device, dtype=dtype) * scale
    return LowRankState(b=b, a=a)


def init_factors_from_observed_svd(
    problem: LowRankCompletionProblem,
    *,
    rank: int,
    seed: int,
    device: torch.device,
    oversampling: int,
    niter: int,
) -> LowRankState:
    observed = torch.zeros((problem.m, problem.n), dtype=DEFAULT_DTYPE, device=device)
    counts = torch.zeros((problem.m, problem.n), dtype=DEFAULT_DTYPE, device=device)
    observed.index_put_((problem.rows, problem.cols), problem.values, accumulate=True)
    counts.index_put_((problem.rows, problem.cols), torch.ones_like(problem.values), accumulate=True)
    observed = observed / counts.clamp_min(1.0)
    q = min(min(problem.m, problem.n), max(rank + int(oversampling), rank))
    torch.manual_seed(seed + 91_337)
    u, s, v = torch.svd_lowrank(observed, q=q, niter=int(niter))
    u_r = u[:, :rank]
    s_r = s[:rank].clamp_min(1e-16)
    v_r = v[:, :rank]
    sqrt_s = torch.sqrt(s_r)
    b = u_r * sqrt_s.unsqueeze(0)
    a = sqrt_s.unsqueeze(1) * v_r.transpose(0, 1)
    return LowRankState(b=b.contiguous(), a=a.contiguous())


def relative_recovery_error_lowrank(state: LowRankState, problem: LowRankCompletionProblem) -> float:
    gram_b, gram_a = factor_grams(state)
    norm_x_sq = torch.sum(gram_b * gram_a.transpose(-1, -2))
    uTb = problem.u_star.transpose(-1, -2) @ state.b
    aV = state.a @ problem.v_star
    cross = torch.sum(problem.sigmas * torch.diagonal(uTb @ aV))
    err_sq = (norm_x_sq + torch.sum(problem.sigmas * problem.sigmas) - 2.0 * cross).clamp_min(0.0)
    return float(torch.sqrt(err_sq).item())


def loss_and_factor_grads(
    problem: LowRankCompletionProblem,
    state: LowRankState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def run_one_training(
    *,
    problem: LowRankCompletionProblem,
    init_state: LowRankState,
    method: str,
    base_lr: float,
    max_iters: int,
    schedule: str,
    log_every: int,
) -> tuple[LowRankState, list[dict[str, Any]], dict[str, Any]]:
    cfg = METHODS[method]
    state = LowRankState(b=init_state.b.clone(), a=init_state.a.clone())
    logs: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for iteration in range(1, max_iters + 1):
        loss, egrad_b, egrad_a = loss_and_factor_grads(problem, state)
        xi_b, xi_a = cfg["step_fn"](state, egrad_b, egrad_a)
        lr_t = float(base_lr)
        if schedule == "sqrt":
            lr_t = lr_t / math.sqrt(iteration)
        elif schedule == "linear":
            lr_t = lr_t / float(iteration)
        elif schedule != "constant":
            raise ValueError(f"Unknown schedule: {schedule}")
        metrics = method_metrics(state, egrad_b, egrad_a, xi_b, xi_a, cfg["norm"], cfg["geometry"])
        state = retract_factors(state, xi_b, xi_a, lr_t)
        should_log = iteration == 1 or iteration == max_iters or iteration % max(1, log_every) == 0
        if should_log:
            logs.append(
                {
                    "iteration": iteration,
                    "base_lr": float(base_lr),
                    "lr_t": float(lr_t),
                    "train_loss": float(loss.item()),
                    "relative_error": relative_recovery_error_lowrank(state, problem),
                    "wall_time_sec": time.perf_counter() - t0,
                    **metrics,
                }
            )
    if not logs or logs[-1]["iteration"] != max_iters:
        loss, egrad_b, egrad_a = loss_and_factor_grads(problem, state)
        xi_b, xi_a = cfg["step_fn"](state, egrad_b, egrad_a)
        metrics = method_metrics(state, egrad_b, egrad_a, xi_b, xi_a, cfg["norm"], cfg["geometry"])
        logs.append(
            {
                "iteration": max_iters,
                "base_lr": float(base_lr),
                "lr_t": float(base_lr) / math.sqrt(max_iters) if schedule == "sqrt" else float(base_lr),
                "train_loss": float(loss.item()),
                "relative_error": relative_recovery_error_lowrank(state, problem),
                "wall_time_sec": time.perf_counter() - t0,
                **metrics,
            }
        )
    final = logs[-1]
    summary = {
        "method": method,
        "base_lr": float(base_lr),
        "schedule": schedule,
        "final_train_loss": final["train_loss"],
        "final_relative_error": final["relative_error"],
        "final_rgrad_norm": final["rgrad_norm"],
        "final_dual_norm_H": final["dual_norm_H"],
        "final_Z_norm_sq": final["Z_norm_sq"],
        "final_kappa_B": final["kappa_B"],
        "final_kappa_A": final["kappa_A"],
        "final_factor_norm_sq": final["factor_norm_sq"],
        "final_wall_time_sec": final["wall_time_sec"],
    }
    return state, logs, summary


def save_run_outputs(
    run_dir: Path,
    *,
    config: dict[str, Any],
    logs: list[dict[str, Any]],
    final_state: LowRankState,
    summary: dict[str, Any],
) -> None:
    ensure_dir(run_dir)
    save_json(run_dir / "config.json", config)
    write_csv(run_dir / "metrics.csv", logs)
    torch.save({"b": final_state.b.detach().cpu(), "a": final_state.a.detach().cpu()}, run_dir / "final_model.pt")
    save_json(run_dir / "summary.json", summary)


def aggregate_results(
    results_dir: Path,
    *,
    size_spec: str,
    r_star: int,
    rank: int,
    kappa_values: list[float],
    snr_db_values: list[float],
    seeds: list[int],
    methods: list[str],
    lrs: list[float],
    schedule: str,
) -> None:
    lr_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    best_lr_payload: dict[str, Any] = {}
    for kappa in kappa_values:
        for snr_db in snr_db_values:
            cell_key = f"completion|{size_spec}|rstar={r_star}|rank={rank}|kappa={kappa}|snr={snr_db}|alpha=1.0"
            best_lr_payload[cell_key] = {}
            for method in methods:
                candidates: list[dict[str, Any]] = []
                for lr in lrs:
                    per_seed = []
                    for seed in seeds:
                        summary_path = (
                            results_dir
                            / f"size_{size_spec}"
                            / f"rstar_{r_star}"
                            / f"rank_{rank}"
                            / f"kappa_{kappa_label(kappa)}"
                            / f"snr_{snr_label(snr_db)}"
                            / "alpha_1"
                            / f"seed_{seed}"
                            / method
                            / f"lr_{format_lr(lr)}"
                            / "summary.json"
                        )
                        if summary_path.exists():
                            import json

                            per_seed.append(json.loads(summary_path.read_text()))
                    if not per_seed:
                        continue
                    rel_errors = [float(item["final_relative_error"]) for item in per_seed]
                    losses = [float(item["final_train_loss"]) for item in per_seed]
                    row = {
                        "variant": "completion",
                        "size": size_spec,
                        "r_star": r_star,
                        "rank": rank,
                        "kappa": float(kappa),
                        "snr_db": "inf" if math.isinf(float(snr_db)) else float(snr_db),
                        "init_gauge_alpha": 1.0,
                        "method": method,
                        "base_lr": float(lr),
                        "schedule": schedule,
                        "n_seeds": len(per_seed),
                        "mean_final_relative_error": float(np.mean(rel_errors)),
                        "std_final_relative_error": float(np.std(rel_errors, ddof=1)) if len(rel_errors) > 1 else 0.0,
                        "mean_final_train_loss": float(np.mean(losses)),
                    }
                    lr_rows.append(row)
                    candidates.append(row)
                if not candidates:
                    continue
                best = min(candidates, key=lambda row: (row["mean_final_relative_error"], row["mean_final_train_loss"], row["base_lr"]))
                best_lr_payload[cell_key][method] = best["base_lr"]
                for seed in seeds:
                    summary_path = (
                        results_dir
                        / f"size_{size_spec}"
                        / f"rstar_{r_star}"
                        / f"rank_{rank}"
                        / f"kappa_{kappa_label(kappa)}"
                        / f"snr_{snr_label(snr_db)}"
                        / "alpha_1"
                        / f"seed_{seed}"
                        / method
                        / f"lr_{format_lr(best['base_lr'])}"
                        / "summary.json"
                    )
                    if summary_path.exists():
                        import json

                        selected_rows.append(
                            {
                                "variant": "completion",
                                "size": size_spec,
                                "r_star": r_star,
                                "rank": rank,
                                "kappa": float(kappa),
                                "snr_db": "inf" if math.isinf(float(snr_db)) else float(snr_db),
                                "init_gauge_alpha": 1.0,
                                "seed": seed,
                                "method": method,
                                "local_best_base_lr": float(best["base_lr"]),
                                **json.loads(summary_path.read_text()),
                            }
                        )
    write_csv(results_dir / "lr_sweep_summary.csv", lr_rows)
    save_json(results_dir / "global_best_lr.json", best_lr_payload)
    write_csv(results_dir / "global_selected_summary.csv", selected_rows)

    mean_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in selected_rows:
        key = (row["kappa"], row["snr_db"], row["method"])
        grouped.setdefault(key, []).append(row)
    for (kappa, snr_db, method), rows in grouped.items():
        rel_errors = [float(row["final_relative_error"]) for row in rows]
        mean_rows.append(
            {
                "variant": "completion",
                "size": size_spec,
                "r_star": r_star,
                "rank": rank,
                "kappa": kappa,
                "snr_db": snr_db,
                "init_gauge_alpha": 1.0,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_seeds": len(rows),
                "selected_base_lr": float(rows[0]["local_best_base_lr"]),
                "schedule": schedule,
                "mean_final_relative_error": float(np.mean(rel_errors)),
                "std_final_relative_error": float(np.std(rel_errors, ddof=1)) if len(rel_errors) > 1 else 0.0,
                "mean_final_train_loss": float(np.mean([float(row["final_train_loss"]) for row in rows])),
                "mean_final_wall_time_sec": float(np.mean([float(row["final_wall_time_sec"]) for row in rows])),
                "mean_final_kappa_B": float(np.mean([float(row["final_kappa_B"]) for row in rows])),
                "mean_final_kappa_A": float(np.mean([float(row["final_kappa_A"]) for row in rows])),
            }
        )
    write_csv(results_dir / "mean_relative_error_table.csv", mean_rows)

    indexed = {(row["kappa"], row["method"]): row for row in mean_rows}
    pair_rows: list[dict[str, Any]] = []
    for kappa in kappa_values:
        for norm_name, intrinsic, euclidean in PAIR_SPECS:
            i_row = indexed.get((float(kappa), intrinsic))
            e_row = indexed.get((float(kappa), euclidean))
            if not i_row or not e_row:
                continue
            i_mean = float(i_row["mean_final_relative_error"])
            e_mean = float(e_row["mean_final_relative_error"])
            pair_rows.append(
                {
                    "norm": norm_name,
                    "kappa": float(kappa),
                    "intrinsic_method": METHOD_LABELS[intrinsic],
                    "euclidean_method": METHOD_LABELS[euclidean],
                    "intrinsic_mean_error": i_mean,
                    "intrinsic_std_error": float(i_row["std_final_relative_error"]),
                    "euclidean_mean_error": e_mean,
                    "euclidean_std_error": float(e_row["std_final_relative_error"]),
                    "intrinsic_minus_euclidean": i_mean - e_mean,
                    "intrinsic_over_euclidean": i_mean / e_mean if e_mean else float("nan"),
                }
            )
    write_csv(results_dir / "pairwise_summary.csv", pair_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large balanced fixed-rank completion with decaying stepsizes.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--size-spec", default="5000x5000")
    parser.add_argument("--rstar", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--kappa-values", nargs="+", type=float, default=[1.0, 10.0, 100.0, 1000.0, 10000.0])
    parser.add_argument("--snr-db-values", nargs="+", default=["inf"])
    parser.add_argument("--seed-values", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--completion-multiplier", type=float, default=5.0)
    parser.add_argument("--schedule", choices=["constant", "sqrt", "linear"], default="sqrt")
    parser.add_argument("--init-mode", choices=["random", "observed-svd"], default="random")
    parser.add_argument("--init-product-norm", type=float, default=1.0)
    parser.add_argument("--svd-oversampling", type=int, default=8)
    parser.add_argument("--svd-niter", type=int, default=4)
    parser.add_argument("--legacy-init-scale", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(int(args.seed_values[0]))
    ensure_dir(args.results_dir)
    device = torch.device(args.device)
    m, n = parse_size_spec(args.size_spec)
    snr_db_values = [parse_snr(token) for token in args.snr_db_values]
    run_config = {
        "variant": "completion",
        "size_spec": args.size_spec,
        "rstar": args.rstar,
        "rank": args.rank,
        "kappa_values": args.kappa_values,
        "snr_db_values": ["inf" if math.isinf(v) else v for v in snr_db_values],
        "seed_values": args.seed_values,
        "methods": args.methods,
        "lrs": args.lrs,
        "max_iters": args.max_iters,
        "completion_multiplier": args.completion_multiplier,
        "init_gauge_alpha": 1.0,
        "init_mode": args.init_mode,
        "init_product_norm": None if args.legacy_init_scale else args.init_product_norm,
        "svd_oversampling": args.svd_oversampling,
        "svd_niter": args.svd_niter,
        "schedule": args.schedule,
        "log_every": args.log_every,
        "device": str(device),
        "dtype": str(DEFAULT_DTYPE),
        "hardware": hardware_info(),
    }
    save_json(args.results_dir / "run_config.json", run_config)

    for kappa in args.kappa_values:
        for snr_db in snr_db_values:
            for seed in args.seed_values:
                problem = build_completion_problem(
                    m=m,
                    n=n,
                    r_star=args.rstar,
                    kappa=float(kappa),
                    seed=seed,
                    completion_multiplier=args.completion_multiplier,
                    snr_db=float(snr_db),
                    device=device,
                )
                if args.init_mode == "observed-svd":
                    init_state = init_factors_from_observed_svd(
                        problem,
                        rank=args.rank,
                        seed=seed,
                        device=device,
                        oversampling=args.svd_oversampling,
                        niter=args.svd_niter,
                    )
                else:
                    init_state = init_factors(
                        m=m,
                        n=n,
                        rank=args.rank,
                        seed=seed,
                        device=device,
                        product_norm=None if args.legacy_init_scale else float(args.init_product_norm),
                    )
                shared_dir = (
                    args.results_dir
                    / f"size_{args.size_spec}"
                    / f"rstar_{args.rstar}"
                    / f"rank_{args.rank}"
                    / f"kappa_{kappa_label(kappa)}"
                    / f"snr_{snr_label(snr_db)}"
                    / "alpha_1"
                    / f"seed_{seed}"
                    / "shared"
                )
                ensure_dir(shared_dir)
                torch.save(
                    {
                        "u_star": problem.u_star.detach().cpu(),
                        "sigmas": problem.sigmas.detach().cpu(),
                        "v_star": problem.v_star.detach().cpu(),
                        "rows": problem.rows.detach().cpu(),
                        "cols": problem.cols.detach().cpu(),
                        "values": problem.values.detach().cpu(),
                    },
                    shared_dir / "completion_lowrank_problem.pt",
                )
                torch.save({"b": init_state.b.detach().cpu(), "a": init_state.a.detach().cpu()}, shared_dir / "init_factors.pt")
                for method in args.methods:
                    for lr in args.lrs:
                        run_dir = (
                            args.results_dir
                            / f"size_{args.size_spec}"
                            / f"rstar_{args.rstar}"
                            / f"rank_{args.rank}"
                            / f"kappa_{kappa_label(kappa)}"
                            / f"snr_{snr_label(snr_db)}"
                            / "alpha_1"
                            / f"seed_{seed}"
                            / method
                            / f"lr_{format_lr(lr)}"
                        )
                        if (run_dir / "summary.json").exists() and not args.overwrite:
                            continue
                        config = {
                            "variant": "completion",
                            "size": args.size_spec,
                            "m": m,
                            "n": n,
                            "r_star": args.rstar,
                            "rank": args.rank,
                            "kappa": float(kappa),
                            "snr_db": "inf" if math.isinf(float(snr_db)) else float(snr_db),
                            "seed": seed,
                            "method": method,
                            "base_lr": float(lr),
                            "schedule": args.schedule,
                            "max_iters": args.max_iters,
                            "init_gauge_alpha": 1.0,
                            "init_mode": args.init_mode,
                            "init_product_norm": None if args.legacy_init_scale else float(args.init_product_norm),
                            "svd_oversampling": args.svd_oversampling,
                            "svd_niter": args.svd_niter,
                            "shared_dir": str(shared_dir),
                        }
                        final_state, logs, summary = run_one_training(
                            problem=problem,
                            init_state=init_state,
                            method=method,
                            base_lr=float(lr),
                            max_iters=int(args.max_iters),
                            schedule=args.schedule,
                            log_every=int(args.log_every),
                        )
                        save_run_outputs(run_dir, config=config, logs=logs, final_state=final_state, summary=summary)

    aggregate_results(
        args.results_dir,
        size_spec=args.size_spec,
        r_star=args.rstar,
        rank=args.rank,
        kappa_values=args.kappa_values,
        snr_db_values=snr_db_values,
        seeds=args.seed_values,
        methods=args.methods,
        lrs=args.lrs,
        schedule=args.schedule,
    )
    print(f"Wrote {args.results_dir / 'mean_relative_error_table.csv'}")
    print(f"Wrote {args.results_dir / 'pairwise_summary.csv'}")


if __name__ == "__main__":
    main()
