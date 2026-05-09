from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import format_lr


DEFAULT_RESULTS_DIR = (
    ROOT
    / "results"
    / "cifar100_fixed_rank_head_formal_valselect_s0s1s2s3s4_r10_b128_e200"
)

METHODS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-select LR/checkpoint for CIFAR-100 fixed-rank head runs."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--lr-grid", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    return parser.parse_args()


def load_metrics(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        for method in args.methods:
            for lr in args.lr_grid:
                path = (
                    args.results_dir
                    / f"seed{seed}"
                    / f"rank{args.rank}"
                    / method
                    / f"lr_{format_lr(lr)}"
                    / "metrics.csv"
                )
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                df["seed"] = seed
                df["rank"] = args.rank
                df["method"] = method
                df["lr"] = float(lr)
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No metrics.csv files found under {args.results_dir}")
    return pd.concat(frames, ignore_index=True)


def _std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


def build_sweep_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["rank", "method", "lr", "epoch"], as_index=False)
    return grouped.agg(
        n_seeds=("seed", "nunique"),
        mean_val_acc=("val_acc", "mean"),
        std_val_acc=("val_acc", _std),
        mean_val_loss=("val_loss", "mean"),
        std_val_loss=("val_loss", _std),
        mean_test_acc=("test_acc", "mean"),
        std_test_acc=("test_acc", _std),
        mean_test_loss=("test_loss", "mean"),
        std_test_loss=("test_loss", _std),
        mean_train_acc=("train_acc", "mean"),
        mean_train_loss=("train_loss", "mean"),
        mean_wall_time_sec=("wall_time_sec", "mean"),
    )


def select_checkpoints(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in sweep.groupby("method"):
        chosen = (
            sub.sort_values(
                ["mean_val_acc", "mean_val_loss", "n_seeds", "epoch", "lr"],
                ascending=[False, True, False, True, True],
            )
            .iloc[0]
            .to_dict()
        )
        chosen["method_label"] = METHOD_LABELS.get(method, method)
        rows.append(chosen)
    order = {method: idx for idx, method in enumerate(METHODS)}
    out = pd.DataFrame(rows)
    return out.sort_values("method", key=lambda col: col.map(order)).reset_index(drop=True)


def build_pairwise(selected: pd.DataFrame) -> pd.DataFrame:
    indexed = selected.set_index("method")
    rows = []
    for pair, intrinsic, euclidean in PAIR_SPECS:
        if intrinsic not in indexed.index or euclidean not in indexed.index:
            continue
        irow = indexed.loc[intrinsic]
        erow = indexed.loc[euclidean]
        rows.append(
            {
                "pair": pair,
                "intrinsic_method": METHOD_LABELS[intrinsic],
                "euclidean_method": METHOD_LABELS[euclidean],
                "intrinsic_lr": float(irow["lr"]),
                "intrinsic_epoch": int(irow["epoch"]),
                "euclidean_lr": float(erow["lr"]),
                "euclidean_epoch": int(erow["epoch"]),
                "intrinsic_test_acc": float(irow["mean_test_acc"]),
                "euclidean_test_acc": float(erow["mean_test_acc"]),
                "test_acc_gap_intrinsic_minus_euclidean": float(irow["mean_test_acc"] - erow["mean_test_acc"]),
                "intrinsic_val_acc": float(irow["mean_val_acc"]),
                "euclidean_val_acc": float(erow["mean_val_acc"]),
                "val_acc_gap_intrinsic_minus_euclidean": float(irow["mean_val_acc"] - erow["mean_val_acc"]),
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(selected: pd.DataFrame, path: Path) -> None:
    display = selected.copy()
    display["Method"] = display["method"].map(METHOD_LABELS).fillna(display["method"])
    display["Selected LR"] = display["lr"].map(lambda x: f"{float(x):g}")
    display["Selected epoch"] = display["epoch"].astype(int).astype(str)
    display["Test acc."] = [
        f"{mean:.3f} $\\pm$ {std:.3f}"
        for mean, std in zip(display["mean_test_acc"], display["std_test_acc"])
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CIFAR-100 fixed-rank linear-probe classification with validation-selected learning rate and checkpoint.}",
        r"\label{tab:cifar-fixedrank-head-valselect}",
        r"{\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Selected LR & Selected epoch & Test acc. \\",
        r"\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(f"{row['Method']} & {row['Selected LR']} & {row['Selected epoch']} & {row['Test acc.']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args)
    sweep = build_sweep_summary(metrics)
    selected = select_checkpoints(sweep)
    pairwise = build_pairwise(selected)

    sweep.to_csv(args.results_dir / "valselect_sweep_summary.csv", index=False)
    selected.to_csv(args.results_dir / "valselect_mean_test_accuracy_table.csv", index=False)
    pairwise.to_csv(args.results_dir / "valselect_pairwise_gap_summary.csv", index=False)
    write_latex_table(selected, args.results_dir / "valselect_table.tex")

    print(f"Wrote {args.results_dir / 'valselect_sweep_summary.csv'}")
    print(f"Wrote {args.results_dir / 'valselect_mean_test_accuracy_table.csv'}")
    print(f"Wrote {args.results_dir / 'valselect_pairwise_gap_summary.csv'}")
    print(selected[["method_label", "lr", "epoch", "mean_val_acc", "mean_test_acc", "std_test_acc"]].to_string(index=False))
    if not pairwise.empty:
        print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
