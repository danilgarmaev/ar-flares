#!/usr/bin/env bash
#SBATCH --job-name=arfl-a1res-arr
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-39

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
# A1 sweep grid (2 backbones x 4 resolutions x 5 seeds = 40 tasks)
# ---------------------------
BACKBONES=(resnet50 efficientnet_b0)
RESOLUTIONS=(224 112 56 28)
N_SEEDS=5

idx=${SLURM_ARRAY_TASK_ID}
seed=$(( idx % N_SEEDS ))
tmp=$(( idx / N_SEEDS ))
res_idx=$(( tmp % ${#RESOLUTIONS[@]} ))
bb_idx=$(( tmp / ${#RESOLUTIONS[@]} ))

BACKBONE="${BACKBONES[$bb_idx]}"
RESOLUTION="${RESOLUTIONS[$res_idx]}"
EPOCHS="${EPOCHS:-50}"
PRETRAINED="${PRETRAINED:-0}"

echo "Task ${SLURM_ARRAY_TASK_ID}: backbone=${BACKBONE} resolution=${RESOLUTION} seed=${seed} offline=${OFFLINE} pretrained=${PRETRAINED}"

cmd=(python -m classifier_NN.legacy.run_experiments_A1 \
  --single \
  --backbone "$BACKBONE" \
  --resolution "$RESOLUTION" \
  --seed "$seed" \
  --epochs "$EPOCHS")

if [[ "$PRETRAINED" == "1" ]]; then
  cmd+=(--pretrained)
else
  cmd+=(--no-pretrained)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
