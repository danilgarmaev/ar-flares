#!/usr/bin/env bash
#SBATCH --job-name=arfl-16f-lm-ds
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5

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

# Offline mode
export OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Put caches on scratch
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ------------------------------------------------------
# 6 runs total for ONE model:
#   C+ @ 36/72/108 with NONE target (default 200k)
#   M+ @ 36/72/108 with train_ds-like NONE target (100k)
#
# Defaults are tuned for R(2+1)D at 112. For VideoMAE, set:
#   MODEL=videomae IMAGE_SIZE=224
# ------------------------------------------------------

MODEL="${MODEL:-r2plus1d_18}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
PRETRAINED_3D="${PRETRAINED_3D:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-112}"
USE_AUG="${USE_AUG:-0}"

# User-adjustable downsampling targets
CPLUS_NONE_TARGET="${CPLUS_NONE_TARGET:-200000}"
MPLUS_NONE_TARGET="${MPLUS_NONE_TARGET:-100000}"

# Arrays for 6 tasks
CADENCES=(36 72 108 36 72 108)
MIN_CLASSES=(C C C M M M)

idx=${SLURM_ARRAY_TASK_ID}
CADENCE="${CADENCES[$idx]}"
MIN_CLASS="${MIN_CLASSES[$idx]}"

if [[ "$MIN_CLASS" == "C" ]]; then
    TARGET_NEG_NONE="$CPLUS_NONE_TARGET"
    NEG_KEEP_PROB_C=""
else
    TARGET_NEG_NONE="$MPLUS_NONE_TARGET"
    # Keep all C negatives for M+ (hard negatives), downsample NONE only.
    NEG_KEEP_PROB_C="1.0"
fi

AUG_FLAG="--no-aug"
if [[ "$USE_AUG" == "1" ]]; then
    AUG_FLAG="--use-aug"
fi

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} cadence=${CADENCE}min min_class=${MIN_CLASS}+ img=${IMAGE_SIZE} seed=${SEED}"
echo "Downsampling: target_neg_none=${TARGET_NEG_NONE} ${NEG_KEEP_PROB_C:+neg_keep_prob_c=${NEG_KEEP_PROB_C}}"
echo "=========================================="

cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence
  --backbone "$MODEL"
  --interval-min "$CADENCE"
  --seed "$SEED"
  --single
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --image-size "$IMAGE_SIZE"
  --lr "$LR"
  --min-flare-class "$MIN_CLASS"
  --balance-classes
  --auto-set-neg-keep-probs
  --target-neg-none "$TARGET_NEG_NONE"
  --loss-type ce
  "$AUG_FLAG")

if [[ "$PRETRAINED_3D" == "1" ]]; then
  cmd+=(--pretrained-3d)
fi

if [[ -n "$NEG_KEEP_PROB_C" ]]; then
  cmd+=(--neg-keep-prob-c "$NEG_KEEP_PROB_C")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} complete!"
echo "=========================================="
