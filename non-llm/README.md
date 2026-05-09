# Anonymous Non-LLM Experiment Code

This archive contains the code needed to reproduce the non-LLM manifold experiments reported in the appendix. It is intentionally code-only: no raw datasets, cached features, result CSVs, figures, logs, local paths, or author-identifying metadata are included.

## Contents

- `src/`: manifold geometry, LMO directions, objectives, metrics, and shared utilities.
- `experiments/`: entry points for the non-LLM experiments.
- `reporting/`: scripts used to aggregate CSV outputs and build convergence plots.
- `tests/`: lightweight geometry and objective tests.
- `environment.yml` and `requirements.txt`: CPU-friendly Python dependencies.

The package covers the following experiments:

- Fixed-rank synthetic matrix completion: `experiments/run_synthetic_completion_balanced_large.py`
- Fixed-rank CIFAR-100 classification heads: `experiments/run_cifar100_fixed_rank_head.py` and `experiments/run_cifar100_fixed_rank_head_selected_lrs.py`
- MovieLens-1M fixed-rank matrix completion: `experiments/run_movielens_fixed_rank.py`
- SPD covariance-based CIFAR-100 classification: `experiments/run_cifar100_spd_proto.py`
- Stiefel subcenter CIFAR-100 classification: `experiments/run_cifar100_stiefel_classifier.py`
- Grassmann subspace learning: `experiments/run_youtube_orthonormal.py`

The fixed-rank scripts include the six norm-matched methods used in the paper: EGD, RGD, Muon, iMuon, NuMuon, and iMuon-Nu. The CIFAR-100 and MovieLens fixed-rank scripts also include the Spectron baseline. The Stiefel script includes the SPEL baseline. These baselines are implemented inside the corresponding experiment scripts rather than as separate top-level experiment families.

## Environment

Create the conda environment with

```bash
conda env create -f environment.yml
conda activate manifold-lmo-spd-grassmann
```

or install the same dependencies with pip:

```bash
python -m pip install -r requirements.txt
```

The experiments are designed to run on CPU. CUDA or MPS can be used for feature extraction when available, but is not required.

## Quick Checks

Run the lightweight tests:

```bash
python -m pytest tests
```

Run smoke experiments without external datasets:

```bash
python experiments/run_cifar100_fixed_rank_head.py --smoke --results-dir results/smoke_fixed_rank_head
python experiments/run_cifar100_stiefel_classifier.py --smoke --results-dir results/smoke_stiefel
python experiments/run_movielens_fixed_rank.py --synthetic --smoke --results-dir results/smoke_movielens
python experiments/run_youtube_orthonormal.py --synthetic-smoke --smoke --results-dir results/smoke_grassmann
python experiments/run_synthetic_completion_balanced_large.py --results-dir results/smoke_synthetic_completion --size-spec 200x200 --rstar 5 --rank 5 --kappa-values 1 10 --seed-values 0 --methods euclidean_gd riemannian_gd euclidean_muon scaled_muon euclidean_numuon scaled_numuon --lrs 0.1 --max-iters 5 --completion-multiplier 5 --device cpu --overwrite
```

The SPD smoke run uses CIFAR-100 features and will download CIFAR-100 through `torchvision` if the dataset is not already present:

```bash
python experiments/run_cifar100_spd_proto.py --smoke --results-dir results/smoke_spd
```

## Data Notes

Synthetic matrix-completion data are generated procedurally by the script.

CIFAR-100 experiments use the public CIFAR-100 dataset through `torchvision`. Frozen ResNet-18 features are cached under the chosen `--feature-cache-dir`.

MovieLens-1M uses the public `ratings.dat` file. Place it at `data/movielens/ml-1m/ratings.dat`, or pass `--data-path /path/to/ratings.dat`.

Grassmann experiments can be smoke-tested with synthetic subspaces using `--synthetic-smoke`. Full real-data runs require precomputed video features in the format expected by `experiments/run_youtube_orthonormal.py`. The helper `experiments/extract_features.py` provides the ResNet-18 feature extraction entry point.

## Reproducibility

All experiment scripts expose seeds, learning-rate grids, data splits, and output directories through command-line flags. The appendix gives the exact grids and settings used for the reported tables and figures. Outputs are written to the supplied `--results-dir` as CSV and JSON files.

This package contains the core non-LLM experiment code and smoke-test commands. Full paper figures can be regenerated from the CSV outputs produced by the scripts using the reporting utilities. Cached datasets, intermediate results, and exploratory scripts are omitted.
