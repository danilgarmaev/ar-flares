#!/usr/bin/env bash
#SBATCH --job-name=arfl-a1swin-adamw1c
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

# Project venv
source /home/dgarmaev/envs/ar-flares/bin/activate

# Force dataset + results locations
export AR_FLARES_WDS_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_flow
export AR_FLARES_RESULTS_BASE=/home/dgarmaev/scratch/ar-flares/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/home/dgarmaev/scratch/ar-flares/data

# Optional offline mode (OFFLINE=1 forces no-download)
OFFLINE="${OFFLINE:-1}"
if [[ "$OFFLINE" == "1" ]]; then
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
else
  export HF_DATASETS_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
  export HF_HUB_OFFLINE=0
fi

# Put caches on scratch
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ---------------------------
# Swin + AdamW + OneCycle (3 seeds)
# ---------------------------
BACKBONE=swin_tiny_patch4_window7_224
RESOLUTION=224
SEEDS=(0 1 2)
seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}

EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
PRETRAINED="${PRETRAINED:-1}"
AUG_PRESET="${AUG_PRESET:-robust}"

# Optional speed controls
MAX_TRAIN_SHARDS="${MAX_TRAIN_SHARDS:-}"
MAX_VAL_SHARDS="${MAX_VAL_SHARDS:-}"
MAX_TEST_SHARDS="${MAX_TEST_SHARDS:-}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"

echo "Task ${SLURM_ARRAY_TASK_ID}: backbone=${BACKBONE} res=${RESOLUTION} seed=${seed} offline=${OFFLINE} pretrained=${PRETRAINED} lr=${LR} wd=${WEIGHT_DECAY} aug_preset=${AUG_PRESET}"

cmd=(python -m classifier_NN.legacy.run_experiments_A1 \
  --single \
  --backbone "$BACKBONE" \
  --resolution "$RESOLUTION" \
  --seed "$seed" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --optimizer adamw \
  --weight-decay "$WEIGHT_DECAY" \
  --scheduler onecycle \
  --use-aug --aug-preset "$AUG_PRESET" \
  --no-freeze-backbone)

if [[ -n "$MAX_TRAIN_SHARDS" ]]; then
  cmd+=(--max-train-shards "$MAX_TRAIN_SHARDS")
fi
if [[ -n "$MAX_VAL_SHARDS" ]]; then
  cmd+=(--max-val-shards "$MAX_VAL_SHARDS")
fi
if [[ -n "$MAX_TEST_SHARDS" ]]; then
  cmd+=(--max-test-shards "$MAX_TEST_SHARDS")
fi
if [[ -n "$STEPS_PER_EPOCH" ]]; then
  cmd+=(--steps-per-epoch "$STEPS_PER_EPOCH")
fi

if [[ "$PRETRAINED" == "1" ]]; then
  cmd+=(--pretrained)
else
  cmd+=(--no-pretrained)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
