# Third-party components

This directory contains source code redistributed from other projects. Each
subdirectory retains its upstream license. Citing public repositories here
does not violate double-blind anonymity — reviewers routinely see third-party
attribution, and our own identity is not derivable from pointing at them.

## pilancilab/GPT2/

- **Origin:** Stanford Pilanci Lab's GPT-2 LoRA NLG harness, itself derived
  from the original LoRA reference implementation by Microsoft.
- **Purpose here:** provides the GPT-2 Medium + LoRA training loop,
  beam-search decoder, E2E NLG dataset plumbing, and BPE vocab used to
  reproduce Table 1 and Table 5 of the paper.
- **Local modification:** `examples/NLG/src/gpt2_ft_muon.py` is a modified
  copy of the original `gpt2_ft.py` that dispatches to the iMuon / Muon
  optimizer instead of AdamW. A header comment in that file notes the
  modification. No other upstream files were changed.
- **License:** MIT, Copyright (c) 2024 Pilanci Research Group. The verbatim
  upstream LICENSE text is included at `pilancilab/LICENSE`.

## RiemanianFinetune/

- **Origin:** Public implementation of Riemannian fine-tuning for LoRA
  accompanying prior work cited in the paper. Only the optimizer module
  `src/optimizers/RiemannianLoRA.py` is vendored here.
- **Purpose here:** provides the Riemannian-SGD baseline used by
  `run_seed_RieSGD.sh` to reproduce the Riemannion (SGD, no momentum) row
  of Table 5.
- **Local modification:** none. The file is vendored as-is to keep the
  baseline comparable to the published Riemannion reference.
- **License:** at the time of this submission (May 2026), the upstream
  repository does not carry an explicit license file. The file is vendored
  unmodified with full attribution; see `RiemanianFinetune/LICENSE` for the
  good-faith attribution notice.

## e2e-metrics/ (not redistributed here)

The BLEU / NIST / METEOR / ROUGE-L / CIDEr scorer from the public
`tuetschek/e2e-metrics` repository is cloned at reproduction time by
`setup_env.sh` and is not redistributed in this snapshot. Its own license
applies to the cloned copy.

## E2E NLG dataset

The E2E NLG training/validation/test splits under
`pilancilab/GPT2/examples/NLG/data/e2e/` are the public E2E NLG Challenge
dataset, redistributed under its original CC-BY-SA-4.0 license.
