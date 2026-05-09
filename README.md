# Manifold Intrinsic Muon


This repository contains code for the experiments in the accompanying paper.
The two subdirectories are self-contained; each has its own setup, data preparation,
and run instructions in its own `README.md`.

## Contents

- [`imuon/`](imuon/) — **LLM fine-tuning experiments (E2E NLG).**
  Reproduces the iMuon row of Table 1 (bit-exact) and the Table 5
  no-momentum block (Vanilla Muon / iMuon / Riemannion-SGD at their
  respective best learning rates) on GPT-2 Medium + LoRA r=4. See
  [`imuon/README.md`](imuon/README.md) for hardware requirements, a
  three-command quickstart, and the full reproduction grid.

- [`non-llm/`](non-llm/) — **Non-LLM manifold experiments (appendix).**
  Fixed-rank synthetic matrix completion, CIFAR-100 fixed-rank heads,
  MovieLens-1M matrix completion, SPD / Stiefel / Grassmann experiments,
  and the EGD / RGD / Muon / iMuon / NuMuon / iMuon-Nu / Spectron / SPEL
  comparisons. See [`non-llm/README.md`](non-llm/README.md) for the conda
  environment, the experiment entry points, and the CSV aggregation
  scripts under `reporting/`.

The two bundles have separate Python environments and do not share code.
Reviewers can run them independently.
