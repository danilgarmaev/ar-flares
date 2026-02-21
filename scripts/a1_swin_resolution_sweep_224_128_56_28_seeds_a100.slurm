#!/usr/bin/env bash
#SBATCH --job-name=arfl-a1swin-res
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-11

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

# Offline safety switches: fail fast instead of hanging on network retries.
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

# Put caches on scratch so they can be reused across jobs.
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ---------------------------
# Swin-Tiny resolution sweep grid (4 resolutions x 3 seeds = 12 tasks)
# Uses the same training recipe as the best Swin run:
# - pretrained=True
# - full fine-tuning (freeze_backbone=False)
# - robust augmentation
# - lr=1e-4
# ---------------------------
BACKBONE="swin_tiny_patch4_window7_224"
RESOLUTIONS=(224 128 56 28)
N_SEEDS=3

idx=${SLURM_ARRAY_TASK_ID}
seed=$(( idx % N_SEEDS ))
res_idx=$(( idx / N_SEEDS ))

RESOLUTION="${RESOLUTIONS[$res_idx]}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adam_paper}"
SCHEDULER="${SCHEDULER:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
PRETRAINED="${PRETRAINED:-1}"

echo "Task ${SLURM_ARRAY_TASK_ID}: backbone=${BACKBONE} resolution=${RESOLUTION} seed=${seed} offline=${OFFLINE} pretrained=${PRETRAINED}"

cmd=(python -m classifier_NN.legacy.run_experiments_A1 \
  --single \
  --backbone "$BACKBONE" \
  --resolution "$RESOLUTION" \
  --seed "$seed" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --optimizer "$OPTIMIZER" \
  --use-aug \
  --aug-preset robust \
  --no-freeze-backbone)

if [[ -n "$SCHEDULER" ]]; then
  cmd+=(--scheduler "$SCHEDULER")
fi
if [[ -n "$WEIGHT_DECAY" ]]; then
  cmd+=(--weight-decay "$WEIGHT_DECAY")
fi

if [[ "$PRETRAINED" == "1" ]]; then
  cmd+=(--pretrained)
else
  cmd+=(--no-pretrained)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
