#!/usr/bin/env bash
# ============================================================
# Focal-loss + balanced-batch experiments
# ============================================================
# Design:
#   2 label classes  × C+, M+
#   3 cadences       × 36, 72, 108 min
#   2 dataset regimes × full (aug), downsampled (auto, no aug)
#   2 backbones      × r2plus1d_18 (img=112), videomae (img=224)
#   => 2 × 3 × 2 × 2 = 24 array tasks (0-23)
#
# Loss:  binary_focal (gamma=2.0, alpha=0.25) for all runs
# Threshold constraints: precision > 0.15, recall > 0.5  (defaults)
# early_stopping_min_epoch = 3
# ============================================================
#SBATCH --job-name=arfl-focal-bal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-23

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

# --------------------------------------------------------
# Task index layout (0-23):
#   0- 5: r2plus1d_18  C+  full/aug          cadence 36,72,108
#   6-11: r2plus1d_18  M+  full/aug          cadence 36,72,108
#  12-17: videomae     C+  full/aug          cadence 36,72,108
#  18-23: videomae     M+  full/aug          cadence 36,72,108
#
# Within each block of 6:
#  0,1,2  → full dataset, aug enabled
#  3,4,5  → downsampled (auto target_neg_none=100k), aug disabled
# --------------------------------------------------------

MODELS=(
  r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18
  r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18
  videomae videomae videomae videomae videomae videomae
  videomae videomae videomae videomae videomae videomae
)
LABEL_CLASSES=(
  C C C C C C
  M M M M M M
  C C C C C C
  M M M M M M
)
# cadence repeats within each block: 36 72 108 36 72 108
CADENCES=(
  36 72 108 36 72 108
  36 72 108 36 72 108
  36 72 108 36 72 108
  36 72 108 36 72 108
)
# regime: 0=full+aug, 1=downsampled+no-aug  (alternates in triplets)
REGIMES=(
  0 0 0 1 1 1
  0 0 0 1 1 1
  0 0 0 1 1 1
  0 0 0 1 1 1
)

idx=${SLURM_ARRAY_TASK_ID}
MODEL="${MODELS[$idx]}"
LABEL_CLASS="${LABEL_CLASSES[$idx]}"
CADENCE="${CADENCES[$idx]}"
REGIME="${REGIMES[$idx]}"

# Tunable via env
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
PRETRAINED_3D="${PRETRAINED_3D:-1}"

# Image size per backbone
IMAGE_SIZE=112
if [[ "$MODEL" == "videomae" || "$MODEL" == "videoswin" ]]; then
  IMAGE_SIZE=224
fi

echo "=========================================="
echo "Task ${idx}: model=${MODEL} class=${LABEL_CLASS}+"
echo "  cadence=${CADENCE}min  regime=${REGIME}  img=${IMAGE_SIZE}"
echo "=========================================="

cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence
  --backbone    "$MODEL"
  --interval-min "$CADENCE"
  --seed        "$SEED"
  --single
  --epochs      "$EPOCHS"
  --batch-size  "$BATCH_SIZE"
  --image-size  "$IMAGE_SIZE"
  --lr          "$LR"
  --min-flare-class "$LABEL_CLASS"
  --loss-type   binary_focal
  --focal-alpha 0.25
  --early-stopping-min-epoch 3
  --threshold-min-precision  0.15
  --threshold-min-recall     0.5
)

if [[ "$PRETRAINED_3D" == "1" ]]; then
  cmd+=(--pretrained-3d)
fi

if [[ "$REGIME" == "0" ]]; then
  # Full dataset + augmentation
  cmd+=(
    --use-aug
    --no-balance-classes
    --no-balanced-batch-sampling
  )
else
  # Downsampled: auto-target 100k NONE negatives, keep all C negatives as hard negs
  cmd+=(
    --no-aug
    --balance-classes
    --auto-set-neg-keep-probs
    --target-neg-none 100000
    --neg-keep-prob-c 1.0
    --no-balanced-batch-sampling
  )
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "=========================================="
echo "Task ${idx} complete!"
echo "=========================================="
