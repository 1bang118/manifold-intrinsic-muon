from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper_results" / "appendix"


COLORS = {
    "EGD": "#7a7a7a",
    "RGD": "#ff7f0e",
    "Muon": "#7a7a7a",
    "iMuon": "#1f77b4",
    "NuMuon": "#7a7a7a",
    "iMuon-Nu": "#1f77b4",
    "Spectron-style": "#2ca02c",
    "SPEL": "#2ca02c",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def lr_from_dir(path: Path) -> float:
    text = path.name.removeprefix("lr_").replace("p", ".")
    return float(text)


def mean_std_curves(paths: list[Path], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for path in paths:
        rows = read_csv(path)
        x = np.array([as_float(row, x_key) for row in rows], dtype=float)
        y = np.array([as_float(row, y_key) for row in rows], dtype=float)
        series.append((x, y))
    if not series:
        raise ValueError(f"No series for {y_key}")

    min_len = min(len(y) for _, y in series)
    x = series[0][0][:min_len]
    ys = np.vstack([y[:min_len] for _, y in series])
    return x, ys.mean(axis=0), ys.std(axis=0, ddof=1) if ys.shape[0] > 1 else np.zeros(min_len)


def plot_curve(ax, paths: list[Path], label: str, x_key: str, y_key: str) -> None:
    x, mean, std = mean_std_curves(paths, x_key, y_key)
    color = COLORS.get(label, None)
    ax.plot(x, mean, label=label, color=color, linewidth=2.0)
    if len(paths) > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def selected_spd_paths() -> dict[str, list[Path]]:
    root = RESULTS / "cifar100_spd_proto_v1_formal_fullsplit_s0s1s2_b64_e20"
    table = read_csv(root / "mean_test_accuracy_table.csv")
    lr_by_method = {row["method"]: row["selected_global_lr"] for row in table}
    mapping = {
        "EGD": "euclidean_gd",
        "RGD": "riemannian_gd",
        "Muon": "euclidean_muon",
        "iMuon": "spd_muon",
        "NuMuon": "euclidean_numuon",
        "iMuon-Nu": "spd_numuon",
    }
    out: dict[str, list[Path]] = {}
    for label, method in mapping.items():
        lr = lr_by_method[method].replace(".", "p")
        out[label] = sorted(root.glob(f"seed*/{method}/lr_{lr}/metrics.csv"))
    return out


def plot_spd() -> None:
    paths = selected_spd_paths()
    pairs = [("Frobenius", "EGD", "RGD"), ("Spectral", "Muon", "iMuon"), ("Nuclear", "NuMuon", "iMuon-Nu")]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharex=True)
    for col, (title, euclidean, intrinsic) in enumerate(pairs):
        ax = axes[col]
        plot_curve(ax, paths[euclidean], euclidean, "epoch", "train_loss")
        plot_curve(ax, paths[intrinsic], intrinsic, "epoch", "train_loss")
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel("Train objective")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, "figure_spd_convergence_curves")


def select_by_val(root: Path, rank: int, methods: list[str]) -> dict[str, tuple[float, list[Path]]]:
    selected: dict[str, tuple[float, list[Path]]] = {}
    for method in methods:
        grouped: dict[float, list[Path]] = defaultdict(list)
        vals: dict[float, list[float]] = defaultdict(list)
        for summary in root.glob(f"seed*/rank{rank}/{method}/lr_*/summary.json"):
            lr = lr_from_dir(summary.parent)
            payload = json.loads(summary.read_text())
            val = payload.get("final_val_acc", payload.get("final_val_accuracy"))
            if val is None:
                continue
            grouped[lr].append(summary.parent / "metrics.csv")
            vals[lr].append(float(val))
        max_n = max((len(v) for v in vals.values()), default=0)
        candidates = [(np.mean(v), lr) for lr, v in vals.items() if len(v) == max_n]
        if not candidates:
            raise ValueError(f"No candidates for {method} in {root}")
        _, best_lr = max(candidates)
        selected[method] = (best_lr, sorted(grouped[best_lr]))
    return selected


def plot_fixed_rank_cifar() -> None:
    root = RESULTS / "cifar100_fixed_rank_head_spectron_svd_rank40_e50_s0s1s2_locallr"
    method_labels = {
        "euclidean_muon": "Muon",
        "spectron_muon": "Spectron-style",
        "scaled_muon": "iMuon",
    }
    selected = select_by_val(root, 40, list(method_labels))
    print("Fixed-rank CIFAR selected LRs:", {method_labels[k]: v[0] for k, v in selected.items()})

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))
    for method, label in method_labels.items():
        _, paths = selected[method]
        plot_curve(ax, paths, label, "epoch", "train_loss")
    ax.set_ylabel("Train objective")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "figure_fixedrank_cifar_spectron_convergence_curves")


def selected_stiefel_paths() -> dict[str, list[Path]]:
    roots = [
        RESULTS / "cifar100_stiefel_subcenter_cosface_m0p5_s64_ppc4_bs64_s0s1s2_e30_selectedlr",
    ]
    label_methods = {"RGD": "rgd", "iMuon": "imuon", "SPEL": "spel"}
    out: dict[str, list[Path]] = {label: [] for label in label_methods}
    for root in roots:
        for label, method in label_methods.items():
            best: dict[int, tuple[float, Path]] = {}
            for summary in root.glob(f"seed*/{method}/lr_*/summary.json"):
                seed = int(summary.parts[-4].removeprefix("seed"))
                payload = json.loads(summary.read_text())
                val = float(payload["final_val_acc"])
                old = best.get(seed)
                if old is None or val > old[0]:
                    best[seed] = (val, summary.parent / "metrics.csv")
            out[label].extend(path for _, path in sorted(best.values(), key=lambda item: str(item[1])))
    return out


def plot_stiefel() -> None:
    paths = selected_stiefel_paths()
    print("Stiefel selected curve counts:", {k: len(v) for k, v in paths.items()})
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))
    for label in ("RGD", "iMuon", "SPEL"):
        plot_curve(ax, paths[label], label, "epoch", "train_loss")
    ax.set_ylabel("Train objective")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "figure_stiefel_cifar_convergence_curves")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plot_spd()
    plot_fixed_rank_cifar()
    plot_stiefel()
    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()
