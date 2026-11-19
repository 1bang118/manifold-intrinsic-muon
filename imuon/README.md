# GPT-2 Medium on E2E NLG with rank-4 LoRA

The corrected no-momentum iMuon update obtained **70.30 BLEU** at seed 110 and **70.06 ± 0.22 BLEU** across seeds 0, 42 and 110 (sample SD). The archived outputs contain one prediction for each of 630 test inputs per seed. All three use the same multi-reference test set.

| Seed | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
|---:|---:|---:|---:|---:|---:|
| 0 | 69.88 | 8.8159 | 46.55 | 71.12 | 2.5228 |
| 42 | 70.01 | 8.8263 | 46.65 | 71.46 | 2.5186 |
| 110 | 70.30 | 8.8523 | 46.62 | 72.03 | 2.5320 |

## Verify the reported metrics on CPU

From the repository root, on Linux with Python 3.12, Git and Java available:

```bash
cd imuon
python3.12 -m venv .score-env
.score-env/bin/pip install numpy==2.4.3 matplotlib==3.10.8 scikit-image==0.26.0 future==1.0.0
git clone https://github.com/tuetschek/e2e-metrics.git _third_party/e2e-metrics
git -C _third_party/e2e-metrics checkout --detach 1b9c4d87fd0bbb171ea966741cdc012588581ed4
.score-env/bin/python results/e2e/verify.py
```

The compressed references are the E2E NLG Challenge test references grouped by input for the original scorer (CC-BY-SA-4.0). The three prediction files are model outputs from the completed training runs. The verifier decompresses the original scoring references, scores each prediction file, checks all 15 values at scorer precision, and computes the mean and sample SD. The table multiplies BLEU, METEOR and ROUGE-L by 100. Training and scoring hardware can differ.

## Regenerate the predictions on one GPU

From the `imuon` directory, use Linux, a CUDA 12.8-compatible NVIDIA driver, a GPU with at least 40 GB, Java on PATH and disk space for the environment, pretrained weights and full checkpoints (allow 20 GB per run).

```bash
bash setup_env.sh
bash data_prep.sh
.venv/bin/python -m unittest discover -s tests -v
E2E_SMOKE=1 bash run_imuon_v5_nomom.sh
bash run_imuon_v5_nomom.sh
for seed in 0 42; do
    METHOD=imuon SEED="$seed" bash run_seed_experiment.sh
done
```

The headline runner calls the seed runner with seed 110. Full runs train for 26,290 updates, beam-decode the final checkpoint and score the predictions. Each writes checkpoints, predictions, references and `metrics.txt` to `runs/imuon_v5_compact_nomom_seed<seed>/`. The two-update smoke reads back its checkpoint in a separate directory. Both runners require all 48 logical Q/V pairs to update at each step. The archived runs used RTX 6000 Pro GPUs. Numerical results can vary with GPU and PyTorch version.

The corrected recipe uses GPT-2 Medium, standard zero-B LoRA initialization, rank 4, alpha 32, dropout 0.1, LR 0.05, shape-based LR adjustment, five epochs, batch size 8, sequence length 512, weight decay 0.01, label smoothing 0.1 and 500 linear warmup steps. Beam size is 10, maximum generation length 64, length penalty 0.8 and no-repeat n-gram size 4.

## Fixed-rank update and historical controls

For W = BA, the corrected `v5_compact` path whitens each factor gradient using the other factor’s Gram matrix, applies the polar approximation and maps the direction back using triangular solves. Augmented QR adds ridge 1e-6 so the update is defined at zero B. The training recipe uses five Newton–Schulz iterations, zero momentum and Nesterov off. The regression tests compare the SVD partial-polar path with a symmetric-Gram reference and check gauge equivariance in the undamped full-rank case.

`METHOD=imuon_legacy SEED=110 bash run_seed_experiment.sh` reproduces the historical V5 recipe (LR 0.005, archived BLEU 70.74). The original vanilla Muon and Riemannion SGD recipes are available via `METHOD=muon SEED=110 bash run_seed_experiment.sh` and `SEED=110 bash run_seed_RieSGD.sh`.

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
