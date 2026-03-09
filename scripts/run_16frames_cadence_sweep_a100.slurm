#!/usr/bin/env bash
#SBATCH --job-name=arfl-16f-cadence
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

# ---------------------------
# 2 models x 3 cadences = 6 runs
# Models: r2plus1d_18, videomae
# Cadences: 36, 72, 108 minutes
# ---------------------------

MODELS=(r2plus1d_18 r2plus1d_18 r2plus1d_18 videomae videomae videomae)
CADENCES=(36 72 108 36 72 108)

idx=${SLURM_ARRAY_TASK_ID}
MODEL="${MODELS[$idx]}"
CADENCE="${CADENCES[$idx]}"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
PRETRAINED_3D="${PRETRAINED_3D:-1}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} cadence=${CADENCE}min seed=${SEED}"
echo "=========================================="

cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence \
  --backbone "$MODEL" \
  --interval-min "$CADENCE" \
  --seed "$SEED" \
  --single \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR")

if [[ "$PRETRAINED_3D" == "1" ]]; then
  cmd+=(--pretrained-3d)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} complete!"
echo "=========================================="
