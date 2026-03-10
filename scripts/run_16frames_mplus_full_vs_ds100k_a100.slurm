#!/usr/bin/env bash
#SBATCH --job-name=arfl-16f-mplus-2x2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-11

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

source /home/dgarmaev/envs/ar-flares/bin/activate

export AR_FLARES_WDS_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_flow
export AR_FLARES_RESULTS_BASE=/home/dgarmaev/scratch/ar-flares/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/home/dgarmaev/scratch/ar-flares/data

export OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# 12 runs:
# 2 backbones (r2plus1d_18, videomae)
# x 2 dataset regimes (full, ds_none100k)
# x 3 cadences (36, 72, 108)
MODELS=(
  r2plus1d_18 r2plus1d_18 r2plus1d_18
  r2plus1d_18 r2plus1d_18 r2plus1d_18
  videomae videomae videomae
  videomae videomae videomae
)
CADENCES=(36 72 108 36 72 108 36 72 108 36 72 108)
REGIMES=(full full full ds100k ds100k ds100k full full full ds100k ds100k ds100k)

idx=${SLURM_ARRAY_TASK_ID}
MODEL="${MODELS[$idx]}"
CADENCE="${CADENCES[$idx]}"
REGIME="${REGIMES[$idx]}"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
PRETRAINED_3D="${PRETRAINED_3D:-1}"
USE_AUG="${USE_AUG:-0}"

# VideoMAE needs 224; R(2+1)D uses 112.
IMAGE_SIZE=112
if [[ "$MODEL" == "videomae" ]]; then
  IMAGE_SIZE=224
fi

AUG_FLAG="--no-aug"
if [[ "$USE_AUG" == "1" ]]; then
  AUG_FLAG="--use-aug"
fi

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} cadence=${CADENCE} regime=${REGIME} min_class=M+ img=${IMAGE_SIZE}"
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
  --min-flare-class M
  "$AUG_FLAG")

if [[ "$PRETRAINED_3D" == "1" ]]; then
  cmd+=(--pretrained-3d)
fi

if [[ "$REGIME" == "ds100k" ]]; then
  # Downsample NONE to 100k, keep C negatives as hard negatives.
  cmd+=(
    --loss-type ce
    --balance-classes
    --auto-set-neg-keep-probs
    --target-neg-none 100000
    --neg-keep-prob-c 1.0
  )
else
  # Full M+ baseline.
  cmd+=(
    --loss-type ce_weighted
    --no-balance-classes
  )
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} complete!"
echo "=========================================="
