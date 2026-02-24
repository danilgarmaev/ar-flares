#!/usr/bin/env bash
#SBATCH --job-name=arfl-a1swin-scratch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

# Matches what worked in your other project; harmless if unused.
module load gcc opencv

# Project venv
source /home/dgarmaev/envs/ar-flares/bin/activate

# Force dataset + results locations
export AR_FLARES_WDS_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_flow
export AR_FLARES_RESULTS_BASE=/home/dgarmaev/scratch/ar-flares/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/home/dgarmaev/scratch/ar-flares/data

# Offline safety switches (we're not downloading pretrained anyway)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Put caches on scratch (still useful for timm internals)
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ---------------------------
# Swin-Tiny from-scratch @224
# ---------------------------
BACKBONE="swin_tiny_patch4_window7_224"
RESOLUTION=224
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adam_paper}"
SCHEDULER="${SCHEDULER:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
NAME_SUFFIX="${NAME_SUFFIX:-scratch}"

cmd=(python -m classifier_NN.legacy.run_experiments_A1 \
  --single \
  --backbone "$BACKBONE" \
  --resolution "$RESOLUTION" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --optimizer "$OPTIMIZER" \
  --use-aug \
  --aug-preset robust \
  --no-freeze-backbone \
  --no-pretrained \
  --name-suffix "$NAME_SUFFIX")

if [[ -n "$SCHEDULER" ]]; then
  cmd+=(--scheduler "$SCHEDULER")
fi
if [[ -n "$WEIGHT_DECAY" ]]; then
  cmd+=(--weight-decay "$WEIGHT_DECAY")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
