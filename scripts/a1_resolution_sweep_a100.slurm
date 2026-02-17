#!/usr/bin/env bash
#SBATCH --job-name=arfl-a1res
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

# Project venv
source /home/dgarmaev/envs/ar-flares/bin/activate

# Force dataset + results locations (do not rely on heuristics)
export AR_FLARES_WDS_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_flow
export AR_FLARES_RESULTS_BASE=/home/dgarmaev/scratch/ar-flares/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/home/dgarmaev/scratch/ar-flares/data

# Offline safety switches (set OFFLINE=1 to force no-download mode)
OFFLINE="${OFFLINE:-0}"
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
# A1 configuration
# ---------------------------
# Override via: sbatch --export=ALL,BACKBONE=resnet50,RESOLUTION=224,RUNS=5,EPOCHS=50 scripts/a1_resolution_sweep_a100.slurm
BACKBONE="${BACKBONE:-resnet50}"
RESOLUTION="${RESOLUTION:-224}"
RUNS="${RUNS:-5}"
EPOCHS="${EPOCHS:-50}"
SMOKE="${SMOKE:-0}"

cmd=(python -m classifier_NN.legacy.run_experiments_A1 \
  --backbone "$BACKBONE" \
  --resolution "$RESOLUTION" \
  --runs "$RUNS" \
  --epochs "$EPOCHS")

if [[ "$SMOKE" == "1" ]]; then
  cmd+=(--smoke)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
