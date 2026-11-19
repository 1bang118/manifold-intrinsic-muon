#!/bin/bash
# Prepare the E2E NLG dataset and the GPT-2 Medium pretrained checkpoint.
# Idempotent: skips files that already exist.

set -euo pipefail

IMUON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$IMUON_ROOT/.venv}"
PYTHON="$VENV_DIR/bin/python"

NLG_ROOT="$IMUON_ROOT/_third_party/pilancilab/GPT2/examples/NLG"
DATA_DIR="$NLG_ROOT/data/e2e"
VOCAB_DIR="$NLG_ROOT/vocab"
CKPT_DIR="$NLG_ROOT/pretrained_checkpoints"
SRC_DIR="$NLG_ROOT/src"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: venv not found at $VENV_DIR. Run setup_env.sh first." >&2
    exit 1
fi

# ---- 1. GPT-2 Medium pretrained checkpoint ----
mkdir -p "$CKPT_DIR"
CKPT_FILE="$CKPT_DIR/gpt2-medium-pytorch_model.bin"
checkpoint_sha=98c7b0558df2c732799e509a8157d392251b3a6b06e2c72eefb3a00eb10f8318
if [[ -e "$CKPT_FILE" || -L "$CKPT_FILE" ]]; then
    echo "$checkpoint_sha  $CKPT_FILE" | sha256sum -c -
else
    download=$(mktemp "$CKPT_DIR/.gpt2-medium.XXXXXX")
    trap 'rm -f "$download"' EXIT
    wget -q --show-progress -O "$download" \
        https://huggingface.co/openai-community/gpt2-medium/resolve/6dcaa7a952f72f9298047fd5137cd6e4f05f41da/pytorch_model.bin
    echo "$checkpoint_sha  $download" | sha256sum -c -
    mv "$download" "$CKPT_FILE"
    trap - EXIT
fi

# ---- 2. Format raw .txt (context||completion) -> .jsonl ({context, completion}) ----
# This is the input format the BPE encoder expects (it does json.loads per line).
format_split() {
    local split="$1"
    local input="$DATA_DIR/${split}.txt"
    local output="$DATA_DIR/${split}_formatted.jsonl"
    if [ ! -f "$input" ]; then
        echo "ERROR: $input not found" >&2
        exit 1
    fi
    if [ -s "$output" ]; then
        echo "$output already exists (non-empty, reusing)"
        return
    fi
    rm -f "$output"  # remove any 0-byte leftover from a prior failed run
    echo "=== Formatting $split.txt -> ${split}_formatted.jsonl ==="
    "$PYTHON" "$SRC_DIR/format_converting_e2e.py" "$input" "$output"
}

format_split train
format_split valid
format_split test   # also serves as decode reference (test_formatted.jsonl)

# ---- 3. BPE-encode formatted JSONL -> tokenized JSONL (used by training + beam) ----
encode_split() {
    local split="$1"
    local input="$DATA_DIR/${split}_formatted.jsonl"
    local output="$DATA_DIR/${split}.jsonl"
    if [ ! -f "$input" ]; then
        echo "ERROR: $input not found (run format_split first)" >&2
        exit 1
    fi
    if [ -s "$output" ]; then
        echo "$output already exists (non-empty, reusing)"
        return
    fi
    rm -f "$output"  # remove any 0-byte leftover from a prior failed run
    echo "=== Encoding ${split}_formatted.jsonl -> $split.jsonl ==="
    "$PYTHON" "$SRC_DIR/gpt2_encode.py" \
        --vocab "$VOCAB_DIR" \
        --input "$input" \
        --output "$output" \
        --add_bos --add_eos
}

encode_split train
encode_split valid
encode_split test

echo ""
(cd "$NLG_ROOT" && sha256sum -c "$IMUON_ROOT/requirements/e2e_inputs.sha256")
echo "=== data_prep.sh done. Next: bash run_imuon_v5_nomom.sh ==="
ls -lh "$DATA_DIR"/*.jsonl "$CKPT_FILE"
