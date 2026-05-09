#!/bin/bash
# Reproduce the iMuon (ours) row of Table 1: E2E NLG test-set results,
# GPT-2 Medium + LoRA r=4, no optimizer momentum.
#
# Single-GPU run. ~30-60 min on an RTX 6000 / A100 class GPU.
# Expects setup_env.sh and data_prep.sh to have been run first.
#
# Override defaults via env vars:
#   VENV_DIR   path to the python venv      (default: $IMUON_ROOT/.venv)
#   WORK_DIR   training output / ckpt dir   (default: $IMUON_ROOT/runs/imuon_v5_nomom)
#   HF_HOME    HuggingFace cache            (default: $IMUON_ROOT/.cache/hf)

set -euo pipefail

IMUON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="$IMUON_ROOT"

VENV_DIR="${VENV_DIR:-$IMUON_ROOT/.venv}"
WORK_DIR="${WORK_DIR:-$IMUON_ROOT/runs/imuon_v5_nomom}"
HF_HOME="${HF_HOME:-$IMUON_ROOT/.cache/hf}"

PYTHON="$VENV_DIR/bin/python"
PILANCILAB_DIR="$IMUON_ROOT/_third_party/pilancilab/GPT2"
PRETRAINED_CKPT="$PILANCILAB_DIR/examples/NLG/pretrained_checkpoints/gpt2-medium-pytorch_model.bin"
E2E_METRICS_DIR="$IMUON_ROOT/_third_party/e2e-metrics"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: venv not found at $VENV_DIR. Run setup_env.sh first." >&2
    exit 1
fi
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: GPT-2 Medium checkpoint not found at $PRETRAINED_CKPT." >&2
    echo "       Run data_prep.sh first." >&2
    exit 1
fi
if [ ! -f "$PILANCILAB_DIR/examples/NLG/data/e2e/train.jsonl" ]; then
    echo "ERROR: encoded data not found. Run data_prep.sh first." >&2
    exit 1
fi
if [ ! -f "$E2E_METRICS_DIR/measure_scores.py" ]; then
    echo "ERROR: e2e-metrics not found at $E2E_METRICS_DIR." >&2
    echo "       Run setup_env.sh first." >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HF_HOME
export HF_HUB_CACHE="$HF_HOME"

mkdir -p "$WORK_DIR"
cd "$PILANCILAB_DIR"

echo "============================================================"
echo "iMuon V5 (no momentum) | E2E NLG | GPT-2 Medium | LoRA r=4"
echo "LR=5e-3  seed=110  epochs=5"
echo "Node: $(hostname)  Date: $(date)"
echo "WORK_DIR=$WORK_DIR"
echo "============================================================"
nvidia-smi || true

EXP_NAME="imuon_v5_nomom"
MASTER_PORT=$((29500 + RANDOM % 1000))

# ---- 1. Train ----
"$PYTHON" -m torch.distributed.launch --nproc_per_node=1 --master_port="$MASTER_PORT" \
    examples/NLG/src/gpt2_ft_muon.py \
    --train_data ./examples/NLG/data/e2e/train.jsonl \
    --valid_data ./examples/NLG/data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint "$PRETRAINED_CKPT" \
    --platform local \
    --clip 0.0 \
    --lr 0.005 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epislon 1e-6 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 5000 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir "$WORK_DIR" \
    --random_seed 110 \
    --trial_name "$EXP_NAME" \
    --muon_momentum 0.0 \
    --no_muon_nesterov \
    --muon_ns_steps 5 \
    --muon_lora_riemannian_muon \
    --muon_lora_riemannian_adjust_lr \
    --muon_lora_riemannian_variant v5

echo ""
echo "Training complete: $(date)"

# ---- 2. Beam decode ----
FINAL_CKPT="$(ls -t "$WORK_DIR"/model_${EXP_NAME}.*.pt 2>/dev/null | head -1 || true)"
if [ -z "$FINAL_CKPT" ]; then
    echo "ERROR: No checkpoint produced in $WORK_DIR" >&2
    exit 1
fi
echo "Using checkpoint: $FINAL_CKPT"

MASTER_PORT=$((29500 + RANDOM % 1000))
"$PYTHON" -m torch.distributed.launch --nproc_per_node=1 --master_port="$MASTER_PORT" \
    examples/NLG/src/gpt2_beam.py \
    --data ./examples/NLG/data/e2e/test.jsonl \
    --batch_size 8 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint "$FINAL_CKPT" \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir "$WORK_DIR" \
    --output_file "predict_${EXP_NAME}.jsonl"

# ---- 3. Decode beam outputs to ref/pred text ----
"$PYTHON" examples/NLG/src/gpt2_decode.py \
    --vocab ./examples/NLG/vocab \
    --sample_file "$WORK_DIR/predict_${EXP_NAME}.jsonl" \
    --input_file ./examples/NLG/data/e2e/test_formatted.jsonl \
    --output_ref_file "$WORK_DIR/e2e_ref.txt" \
    --output_pred_file "$WORK_DIR/e2e_pred.txt"

# ---- 4. Score with e2e-metrics ----
"$PYTHON" "$E2E_METRICS_DIR/measure_scores.py" \
    "$WORK_DIR/e2e_ref.txt" "$WORK_DIR/e2e_pred.txt" -p > "$WORK_DIR/metrics.txt"

echo ""
echo "============================================================"
cat "$WORK_DIR/metrics.txt"
echo "============================================================"
echo "Done: $(date)"
