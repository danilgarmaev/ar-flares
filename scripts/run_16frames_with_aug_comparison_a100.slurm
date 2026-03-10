#!/usr/bin/env bash
#SBATCH --job-name=arfl-16f-aug-comp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-11

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

# ---------------------------
# Augmentation comparison: with & without
# 2 models x 3 cadences x 2 aug settings = 12 runs
# ---------------------------

# Arrays for configuration
MODELS=(r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18 \
        videomae videomae videomae videomae videomae videomae)
CADENCES=(36 72 108 36 72 108 36 72 108 36 72 108)
USE_AUG=(0 0 0 1 1 1 0 0 0 1 1 1)

idx=${SLURM_ARRAY_TASK_ID}
MODEL="${MODELS[$idx]}"
CADENCE="${CADENCES[$idx]}"
AUG="${USE_AUG[$idx]}"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
PRETRAINED_3D="${PRETRAINED_3D:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-112}"
VIDEOMAE_IMAGE_SIZE="${VIDEOMAE_IMAGE_SIZE:-224}"

if [[ "$MODEL" == "videomae" ]]; then
  IMAGE_SIZE="$VIDEOMAE_IMAGE_SIZE"
fi

AUG_STR="noaug"
if [[ "$AUG" == "1" ]]; then
    AUG_STR="aug"
fi

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} cadence=${CADENCE}min img=${IMAGE_SIZE} aug=${AUG_STR} seed=${SEED}"
echo "=========================================="

cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence \
  --backbone "$MODEL" \
  --interval-min "$CADENCE" \
  --seed "$SEED" \
  --single \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --lr "$LR")

if [[ "$PRETRAINED_3D" == "1" ]]; then
  cmd+=(--pretrained-3d)
fi

if [[ "$AUG" == "1" ]]; then
  cmd+=(--use-aug)
else
  cmd+=(--no-aug)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} complete!"
echo "=========================================="
