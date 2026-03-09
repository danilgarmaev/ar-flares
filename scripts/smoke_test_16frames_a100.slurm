#!/usr/bin/env bash
#SBATCH --job-name=arfl-16f-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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
# Smoke test: R(2+1)D with 72-min cadence (matches macros)
# Quick run with limited epochs/batches to verify setup
# ---------------------------

echo "=========================================="
echo "Smoke Test: R(2+1)D with 72-min cadence"
echo "5 epochs, 100 steps/epoch, 50 val batches"
echo "=========================================="

python -m classifier_NN.legacy.run_experiments_16frames_cadence \
  --backbone r2plus1d_18 \
  --interval-min 72 \
  --seed 0 \
  --single \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-4 \
  --steps-per-epoch 100 \
  --val-max-batches 50 \
  --pretrained-3d

echo ""
echo "=========================================="
echo "Smoke test complete!"
echo "Check results/ for output"
echo "=========================================="
