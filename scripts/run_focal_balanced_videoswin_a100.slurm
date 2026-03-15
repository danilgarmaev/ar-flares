#!/usr/bin/env bash
#SBATCH --job-name=arfl-focal-swin
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

# 12 runs: 2 labels x 3 cadences x 2 regimes for videoswin backbone
MODEL=videoswin
LABEL_CLASSES=(C C C C C C M M M M M M)
CADENCES=(36 72 108 36 72 108 36 72 108 36 72 108)
REGIMES=(0 0 0 1 1 1 0 0 0 1 1 1)  # 0=full+aug, 1=ds100k+noaug

idx=${SLURM_ARRAY_TASK_ID}
LABEL_CLASS="${LABEL_CLASSES[$idx]}"
CADENCE="${CADENCES[$idx]}"
REGIME="${REGIMES[$idx]}"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
IMAGE_SIZE=224

echo "=========================================="
echo "Task ${idx}: model=${MODEL} class=${LABEL_CLASS}+ cadence=${CADENCE}min regime=${REGIME}"
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
  --min-flare-class "$LABEL_CLASS"
  --loss-type binary_focal
  --focal-alpha 0.25
  --early-stopping-min-epoch 3
  --threshold-min-precision 0.15
  --threshold-min-recall 0.5
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

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Task ${idx} complete"
