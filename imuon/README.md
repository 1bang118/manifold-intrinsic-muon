# iMuon: E2E NLG reproduction

This repository reproduces the **iMuon (ours)** row of Table 1 in the
accompanying paper: E2E NLG test-set results with GPT-2 Medium fine-tuned via
LoRA (rank 4) and the iMuon optimizer with no momentum.

| Method        | BLEU  | NIST | METEOR | ROUGE-L | CIDEr |
|---------------|-------|------|--------|---------|-------|
| **iMuon (ours)** | **70.74** | **8.88** | **46.79** | **72.14** | **2.54** |

## Hardware and time

- **GPU:** a single CUDA GPU with ≥40 GB memory (A100, RTX 6000-class, or H100).
  The run was originally produced on an RTX 6000 Pro Blackwell.
- **Wall time:** ~30–60 minutes total (training ≈25 min, beam decode ≈10 min,
  scoring <1 min) on the hardware above. Slower GPUs or smaller memory will
  require lowering `--train_batch_size` in `run_imuon_v5_nomom.sh`.
- **Disk:** ~5 GB (1.5 GB pretrained checkpoint + venv + outputs).

## Reproduce

```bash
git clone <this-repo> imuon && cd imuon
bash setup_env.sh        # install uv, create venv, install pinned deps, clone e2e-metrics
bash data_prep.sh        # download GPT-2 Medium ckpt, BPE-encode E2E data
bash run_imuon_v5_nomom.sh
```

The final command prints the five test-set metrics. Compare against the table
above. Numbers should match within rounding (deterministic seed 110, but minor
hardware-dependent floating-point drift is expected on different GPU models).

Outputs are written to `runs/imuon_v5_nomom/`:

- `model_imuon_v5_nomom.*.pt`  - fine-tuned LoRA checkpoint
- `predict_imuon_v5_nomom.jsonl`  - beam-search outputs
- `e2e_ref.txt`, `e2e_pred.txt`  - reference and prediction text for scoring
- `metrics.txt`  - the five reported metrics

## Full reproduction (Table 5, no-momentum block)

In addition to the headline iMuon row, this artifact can reproduce the three
no-momentum rows of Table 5 (E2E NLG, GPT-2 Medium, LoRA r=4) at any seed.
Each entry below is one training run + beam decode + scoring, ~30–60 minutes.

| Method                 | Learning rate | Runner script              | Target BLEU (seed=110) |
|------------------------|---------------|----------------------------|------------------------|
| Vanilla Muon (no mom.) | 1e-3          | `run_seed_experiment.sh`   | 70.02                  |
| iMuon V5 (no mom.)     | 5e-3          | `run_seed_experiment.sh`   | 70.74                  |
| Riemannion SGD (no mom.) | 5e-2        | `run_seed_RieSGD.sh`       | 70.02                  |

Each runner is parameterized by environment variables and writes to a
seed-stamped subdirectory under `runs/` so concurrent seeds do not collide.

```bash
# Vanilla Muon, seed 42
METHOD=muon  SEED=42  bash run_seed_experiment.sh

# iMuon V5, seed 42
METHOD=imuon SEED=42  bash run_seed_experiment.sh

# Riemannion (SGD), seed 42
SEED=42  bash run_seed_RieSGD.sh
```

Outputs land in `runs/<method>_nomom_seed<SEED>/metrics.txt`. Learning rates
are pinned inside each script to the best-LR row of Table 5 — do not override
unless you are running a sweep. For the published variance study, re-run each
method across seeds {42, 110, 2024} and average the BLEU column.

Reproducibility notes:

- The random seed is pinned, but floating-point non-associativity on different
  GPU models can cause ≤0.05 BLEU drift from the numbers reported above.
  Matches within rounding are expected; exact bit-for-bit match is only
  guaranteed on the RTX 6000 Pro Blackwell the paper was produced on.
- Each run produces ~10 GB of intermediate checkpoints under its work dir
  (one per save interval). Only the last checkpoint is used for the reported
  metric; the rest can be deleted to free disk.
- The Riemannion runner uses a 2r-block parameterization, so its beam-decode
  step loads the checkpoint with `--lora_dim 8 --lora_alpha 64` (effective
  rank is still 4). This is internal to `run_seed_RieSGD.sh` and does not
  require any user action.

## What's in this repository

- `run_imuon_v5_nomom.sh` - single-GPU end-to-end pipeline (train + beam + decode + score)
- `setup_env.sh` - one-time environment bootstrap
- `data_prep.sh` - data + pretrained checkpoint preparation
- `swift/trainers/optimizers/muon.py` - the iMuon optimizer (V1/V5 Riemannian variants, Newton-Schulz iteration)
- `swift/trainers/optimizers/muon_batched.py` - batched variant
- `_third_party/pilancilab/GPT2/` - GPT-2 LoRA NLG harness (vendored from prior work, see attribution below)
- `requirements/lockfiles/e2e_nlg_pinned.txt` - pinned dependency versions

## Attribution

This artifact vendors code from two public upstream projects and clones a
third at reproduction time. Full origin, license, and modification notes are
in [`_third_party/NOTICE.md`](_third_party/NOTICE.md). In brief:

- `_third_party/pilancilab/GPT2/` — GPT-2 LoRA NLG fine-tuning harness,
  adapted from the Stanford Pilanci Lab's public release (itself derived
  from the original LoRA reference implementation). Only
  `examples/NLG/src/gpt2_ft_muon.py` carries a local modification, clearly
  headered in the file.
- `_third_party/RiemanianFinetune/` — Riemannian-SGD LoRA optimizer
  baseline from the public Riemannion reference implementation, vendored
  unmodified; used by `run_seed_RieSGD.sh`.
- `_third_party/e2e-metrics/` — not redistributed here; cloned by
  `setup_env.sh` from the [E2E-NLG-Challenge metrics repository](https://github.com/tuetschek/e2e-metrics).
- The E2E NLG dataset is from the [E2E NLG Challenge](http://www.macs.hw.ac.uk/InteractionLab/E2E/),
  redistributed under its original CC-BY-SA-4.0 license.

Our own code is released under MIT (see [`LICENSE`](LICENSE)). Third-party
components retain their upstream licenses.

## Configuration of the iMuon (ours) run

The configuration that produced the reported numbers is encoded as flag values
in `run_imuon_v5_nomom.sh`. Key choices:

- LoRA: `r=4`, `alpha=32`, `dropout=0.1`
- Schedule: 5 epochs, linear warmup over 500 steps, batch size 8, seq len 512
- iMuon: variant `v5`, Newton-Schulz steps = 5, **momentum = 0** (no Nesterov), Riemannian LR adjustment enabled
- Learning rate: `5e-3`
- Random seed: `110`

Other learning rates (full sweep) and the with-momentum block are deferred to
the supplementary material.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `setup_env.sh`: `uv: command not found` | new shell hasn't picked up `~/.local/bin` | `source ~/.bashrc` or open a new shell and re-run |
| `data_prep.sh`: 401/403 on the GPT-2 checkpoint | transient HF mirror issue | retry; or from inside the venv: `huggingface-cli download openai-community/gpt2-medium pytorch_model.bin` |
| `run_*.sh`: `ImportError: No module named 'optimizers'` | `REPO_DIR` not exported / `swift/trainers/` not on `sys.path` | confirm the runner printed `REPO_DIR=<repo-root>` near the top; if blank, pass `REPO_DIR=$PWD` explicitly |
| CUDA OOM during training or beam decode | GPU has <40 GB memory | lower `--train_batch_size` (train) or `--batch_size` (beam) from 8 to 4 in the runner script |
| `metrics.txt` is empty | `e2e-metrics` not cloned | re-run `setup_env.sh`; verify `_third_party/e2e-metrics/measure_scores.py` exists |
| BLEU off by >1 from the target | flag drift in a local edit | diff the `train` stanza of your runner against the committed version |

