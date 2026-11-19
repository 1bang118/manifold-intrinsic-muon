#!/bin/bash
# One-time environment bootstrap for the iMuon E2E NLG reproduction.
# Idempotent: re-running is safe (skips work that's already done).
#
# Steps:
#   1. install uv (Astral's pip/venv manager) if not on PATH
#   2. create a python 3.12 venv at $IMUON_ROOT/.venv
#   3. install torch 2.11.0+cu128 + triton from the PyTorch cu128 index
#   4. install remaining pinned deps from requirements/lockfiles/e2e_nlg_pinned.txt
#   5. clone tuetschek/e2e-metrics into _third_party/ for BLEU/NIST/METEOR/ROUGE-L/CIDEr scoring

set -euo pipefail

IMUON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$IMUON_ROOT/.venv}"
LOCKFILE="$IMUON_ROOT/requirements/lockfiles/e2e_nlg_pinned.txt"
E2E_METRICS_DIR="$IMUON_ROOT/_third_party/e2e-metrics"

# ---- 1. uv ----
if ! command -v uv >/dev/null 2>&1; then
    if [ ! -x "$HOME/.local/bin/uv" ]; then
        echo "=== Installing uv ==="
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi
UV_BIN="$(command -v uv)"
echo "uv: $UV_BIN ($($UV_BIN --version))"

# ---- 2. venv ----
if [ -d "$VENV_DIR" ]; then
    echo "venv exists at $VENV_DIR (reusing)"
else
    echo "=== Creating venv at $VENV_DIR (python 3.12) ==="
    "$UV_BIN" venv --python 3.12 "$VENV_DIR"
fi

# ---- 3. torch + triton from cu128 index (--no-deps) ----
# Why --no-deps: the lockfile already pins every transitive (fsspec, numpy,
# sympy, networkx, jinja2, the nvidia-*-cu12 cuda runtime wheels, etc.).
# Letting torch's resolver pull them freely would override the lockfile's
# pins and create version conflicts (notably: torch pulls fsspec==2026.2.0
# but datasets==3.2.0 declares fsspec<=2024.9.0 in its metadata).
echo "=== Installing torch 2.11.0+cu128 stack (--no-deps) ==="
"$UV_BIN" pip install --no-deps \
    --python "$VENV_DIR/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
    triton==3.6.0

# ---- 4. remaining deps from lockfile (also --no-deps) ----
# The lockfile is a `uv pip freeze` snapshot of a working install. Each line
# is a pinned version known to coexist at runtime even when individual
# package metadata claims tighter constraints. --no-deps replays the freeze
# verbatim instead of re-resolving (which uv would refuse for the reason
# above).
if [ ! -f "$LOCKFILE" ]; then
    echo "ERROR: lockfile missing at $LOCKFILE" >&2
    exit 1
fi
echo "=== Installing remaining pinned deps (--no-deps) ==="
TMP_REST="$(mktemp)"
grep -Ev '^(#|$|torch(==|$)|torchvision|torchaudio|triton(==|$))' "$LOCKFILE" > "$TMP_REST"
"$UV_BIN" pip install --no-deps --python "$VENV_DIR/bin/python" -r "$TMP_REST"
rm -f "$TMP_REST"

# BPE encoder dependency, additional to the frozen training environment.
"$UV_BIN" pip install --no-deps --python "$VENV_DIR/bin/python" progress==1.6

# Pin the scorer used for the published predictions, including its METEOR jar.
scorer_commit=1b9c4d87fd0bbb171ea966741cdc012588581ed4
if [[ ! -d "$E2E_METRICS_DIR" ]]; then
    git clone https://github.com/tuetschek/e2e-metrics.git "$E2E_METRICS_DIR"
    git -C "$E2E_METRICS_DIR" checkout --detach "$scorer_commit"
fi
if [[ "$(git -C "$E2E_METRICS_DIR" rev-parse HEAD)" != "$scorer_commit" ]]; then
    echo "Expected e2e-metrics revision $scorer_commit at $E2E_METRICS_DIR" >&2
    exit 1
fi
java -version

# ---- 6. quick sanity ----
echo ""
echo "=== Smoke test (versions only; cuda may not be available on a login node) ==="
"$VENV_DIR/bin/python" - <<'PY'
import sys
print(f"python: {sys.version.split()[0]}")
import importlib
for pkg in ("torch", "transformers", "peft", "accelerate", "datasets", "numpy"):
    m = importlib.import_module(pkg)
    print(f"  {pkg}: {getattr(m, '__version__', '?')}")
import torch
print(f"cuda available: {torch.cuda.is_available()}")
PY

echo ""
echo "=== setup_env.sh done. Next: bash data_prep.sh ==="
