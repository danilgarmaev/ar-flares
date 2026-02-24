#!/usr/bin/env bash
#SBATCH --job-name=arfl-a3quick
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

module load gcc opencv
source /home/dgarmaev/envs/ar-flares/bin/activate

# Force dataset + results locations
export AR_FLARES_WDS_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_flow
export AR_FLARES_RESULTS_BASE=/home/dgarmaev/scratch/ar-flares/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/home/dgarmaev/scratch/ar-flares/data

# Offline by default (requires prefetch on login node)
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

export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Exactly 4 runs: one backbone each
BACKBONES=(r3d_18 r2plus1d_18 slowfast timesformer)
BACKBONE="${BACKBONES[$SLURM_ARRAY_TASK_ID]}"

# Single setting knobs
INTERVAL_MIN="${INTERVAL_MIN:-48}"
SEED="${SEED:-0}"
STUDY="${STUDY:-context}"

# Training defaults (override via env)
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SCHEDULER="${SCHEDULER:-cosine}"
PRETRAINED_3D="${PRETRAINED_3D:-1}"

# Optional debug knobs to match older successful runs
BALANCE_CLASSES="${BALANCE_CLASSES:-0}"
NEG_KEEP_PROB="${NEG_KEEP_PROB:-0.25}"
LOSS_TYPE="${LOSS_TYPE:-}"
USE_AUG="${USE_AUG:-}"

IMG_SIZE="${IMG_SIZE:-112}"
T_FRAMES="${T_FRAMES:-16}"
NAME_SUFFIX="${NAME_SUFFIX:-quick4}"  # optional

echo "Task ${SLURM_ARRAY_TASK_ID}: bb=${BACKBONE} interval=${INTERVAL_MIN}min seed=${SEED} study=${STUDY} offline=${OFFLINE}"

cmd=(python -m classifier_NN.legacy.run_experiments_A3 \
  --study "$STUDY" \
  --single \
  --backbone "$BACKBONE" \
  --interval-min "$INTERVAL_MIN" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --optimizer "$OPTIMIZER" \
  --weight-decay "$WEIGHT_DECAY" \
  --scheduler "$SCHEDULER" \
  --img-size "$IMG_SIZE" \
  --T "$T_FRAMES")

if [[ -n "$NAME_SUFFIX" ]]; then
  cmd+=(--name-suffix "$NAME_SUFFIX")
fi

if [[ "$PRETRAINED_3D" == "1" ]]; then
  cmd+=(--pretrained-3d)
else
  cmd+=(--no-pretrained-3d)
fi

if [[ "$BALANCE_CLASSES" == "1" ]]; then
  cmd+=(--balance-classes)
fi

if [[ -n "$LOSS_TYPE" ]]; then
  cmd+=(--loss-type "$LOSS_TYPE")
fi

if [[ -n "$USE_AUG" ]]; then
  if [[ "$USE_AUG" == "1" ]]; then
    cmd+=(--use-aug)
  else
    cmd+=(--no-use-aug)
  fi
fi

if [[ "$BALANCE_CLASSES" == "1" ]]; then
  cmd+=(--neg-keep-prob "$NEG_KEEP_PROB")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
