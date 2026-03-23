#!/usr/bin/env bash
#SBATCH --job-name=arfl-vswin-smoke
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_GPUS="${NUM_GPUS:-2}"
IMAGE_SIZE=224
CADENCE="${CADENCE:-72}"
LABEL_CLASS="${LABEL_CLASS:-C}"
REGIME="${REGIME:-0}"

cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence
  --backbone videoswin
  --interval-min "$CADENCE"
  --seed "$SEED"
  --single
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --image-size "$IMAGE_SIZE"
  --lr "$LR"
  --min-flare-class "$LABEL_CLASS"
  --loss-type binary_focal
  --focal-alpha 0.25
  --early-stopping-min-epoch 1
  --threshold-min-precision 0.15
  --threshold-min-recall 0.5
  --num-workers "$NUM_WORKERS"
  --use-multi-gpu
  --multi-gpu-max-devices "$NUM_GPUS"
)

if [[ "$REGIME" == "0" ]]; then
  cmd+=(
    --use-aug
    --no-balance-classes
    --no-balanced-batch-sampling
  )
else
  cmd+=(
    --no-aug
    --balance-classes
    --auto-set-neg-keep-probs
    --target-neg-none 100000
    --neg-keep-prob-c 1.0
    --no-balanced-batch-sampling
  )
fi

echo "Running smoke test: ${cmd[*]}"
"${cmd[@]}"
