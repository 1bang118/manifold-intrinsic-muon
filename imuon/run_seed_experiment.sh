#!/bin/bash
# Train, beam-decode and score the selected rank-4 E2E recipe.
set -euo pipefail

if [ -z "${METHOD:-}" ] || [ -z "${SEED:-}" ]; then
    echo "ERROR: both METHOD and SEED env vars must be set." >&2
    echo "       e.g. METHOD=imuon SEED=42 bash $0" >&2
    exit 1
fi

IMUON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="$IMUON_ROOT"

VENV_DIR="${VENV_DIR:-$IMUON_ROOT/.venv}"
HF_HOME="${HF_HOME:-$IMUON_ROOT/.cache/hf}"

PYTHON="$VENV_DIR/bin/python"
PILANCILAB_DIR="$IMUON_ROOT/_third_party/pilancilab/GPT2"
PRETRAINED_CKPT="$PILANCILAB_DIR/examples/NLG/pretrained_checkpoints/gpt2-medium-pytorch_model.bin"
E2E_METRICS_DIR="$IMUON_ROOT/_third_party/e2e-metrics"

# Resolve method-specific configuration.
case "$METHOD" in
    muon)
        LR=0.001
        EXP_NAME="muon_nomom_seed${SEED}"
        # Vanilla Muon (no momentum, no Riemannian preconditioning)
        METHOD_FLAGS=(--muon_momentum 0.0 --no_muon_nesterov --muon_ns_steps 5)
        TARGET_BLEU="70.02"
        ;;
    imuon|imuon_legacy)
        variant=v5_compact
        LR=0.05
        TARGET_BLEU=70.30
        if [[ "$METHOD" == imuon_legacy ]]; then
            variant=v5
            LR=0.005
            TARGET_BLEU=70.74
        fi
        EXP_NAME="imuon_${variant}_nomom_seed${SEED}"
        METHOD_FLAGS=(--muon_momentum 0.0 --no_muon_nesterov --muon_ns_steps 5 \
                      --muon_lora_riemannian_muon \
                      --muon_lora_riemannian_adjust_lr \
                      --muon_lora_riemannian_variant "$variant" --muon_lora_precond_eps 1e-6
                      --muon_lora_riemannian_ortho_method ns --expected_lora_pairs 48)
        ;;
    *)
        echo "ERROR: unknown METHOD='$METHOD' (expected: muon | imuon | imuon_legacy)" >&2
        exit 1
        ;;
esac

# A two-update smoke exercises the same model/data/optimizer and saves a checkpoint.

FINAL_STEP=26290
if [[ "${E2E_SMOKE:-0}" == 1 ]]; then
    EXP_NAME="${EXP_NAME}_smoke"
    FINAL_STEP=2
    METHOD_FLAGS+=(--max_step 2 --log_interval 1)
fi
WORK_DIR="${WORK_DIR:-$IMUON_ROOT/runs/$EXP_NAME}"
export PYTHONPATH="$PILANCILAB_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: venv not found at $VENV_DIR. Run setup_env.sh first." >&2
    exit 1
fi
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: GPT-2 Medium checkpoint not found. Run data_prep.sh first." >&2
    exit 1
fi
if [ ! -f "$PILANCILAB_DIR/examples/NLG/data/e2e/train.jsonl" ]; then
    echo "ERROR: encoded data not found. Run data_prep.sh first." >&2
    exit 1
fi
if [ ! -f "$E2E_METRICS_DIR/measure_scores.py" ]; then
    echo "ERROR: e2e-metrics not found. Run setup_env.sh first." >&2
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
echo "Method: $METHOD  | Seed: $SEED  | LR: $LR"
echo "Target BLEU at seed=110: $TARGET_BLEU"
echo "WORK_DIR: $WORK_DIR"
echo "Node: $(hostname)  Date: $(date)"
echo "============================================================"
nvidia-smi || true

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
    --lr "$LR" \
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
    --random_seed "$SEED" \
    --trial_name "$EXP_NAME" \
    "${METHOD_FLAGS[@]}"

echo ""
echo "Training complete: $(date)"

# ---- 2. Beam decode ----
FINAL_CKPT="$WORK_DIR/model_${EXP_NAME}.${FINAL_STEP}.pt"
if [ ! -s "$FINAL_CKPT" ]; then
    echo "ERROR: Missing final checkpoint $FINAL_CKPT" >&2
    exit 1
fi
echo "Using checkpoint: $FINAL_CKPT"
if [[ "${E2E_SMOKE:-0}" == 1 ]]; then
    "$PYTHON" -c 'import sys, torch; assert torch.load(sys.argv[1], map_location="cpu")["model_state_dict"]' "$FINAL_CKPT"
    exit 0
fi

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

# ---- 4. Score ----
"$PYTHON" "$E2E_METRICS_DIR/measure_scores.py" \
    "$WORK_DIR/e2e_ref.txt" "$WORK_DIR/e2e_pred.txt" -p > "$WORK_DIR/metrics.txt"

echo ""
echo "============================================================"
echo "RESULTS — $METHOD | seed=$SEED | lr=$LR"
echo "============================================================"
cat "$WORK_DIR/metrics.txt"
echo "============================================================"
echo "Done: $(date)"
