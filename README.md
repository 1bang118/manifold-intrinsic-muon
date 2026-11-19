# Manifold Intrinsic Muon


This repository contains code for the experiments in the accompanying paper.
The two subdirectories are self-contained; each has its own setup, data preparation,
and run instructions in its own `README.md`.

## Contents

- [`imuon/`](imuon/) — **LLM fine-tuning experiments (E2E NLG).**
  Includes the corrected fixed-rank LoRA optimizer, training recipes for the
  seed-110 result (70.30 BLEU) and three-seed result (70.06 ± 0.22), and the
  predictions needed to verify all five metrics on CPU.
  See the [E2E reproduction instructions](imuon/README.md).

- [`non-llm/`](non-llm/) — **Non-LLM manifold experiments (appendix).**
  Fixed-rank synthetic matrix completion, CIFAR-100 fixed-rank heads,
  MovieLens-1M matrix completion, SPD / Stiefel / Grassmann experiments,
  and the EGD / RGD / Muon / iMuon / NuMuon / iMuon-Nu / Spectron / SPEL
  comparisons. See [`non-llm/README.md`](non-llm/README.md) for the conda
  environment, the experiment entry points, and the CSV aggregation
  scripts under `reporting/`.

The two bundles have separate Python environments and do not share code.
Reviewers can run them independently.
