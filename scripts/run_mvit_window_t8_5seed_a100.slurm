#!/usr/bin/env bash
# 5-seed A3 T=8 MViT temporal-window sweep for C+ and M+ labels.
# Windows: 12h, 24h, 36h, 48h, 60h.
# Usage:
#   sbatch scripts/run_mvit_window_t8_5seed_a100.slurm

#SBATCH --job-name=arfl-mvit-wt8
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-49%10

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results figures

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

WINDOW_HOURS=(12 24 36 48 60)

idx="${SLURM_ARRAY_TASK_ID}"
window_idx=$(( idx / 10 ))
slot=$(( idx % 10 ))

WINDOW_HOUR="${WINDOW_HOURS[$window_idx]}"
if (( slot < 5 )); then
  LABEL_CLASS="C"
  LABEL_TAG="c"
  SEED="${slot}"
else
  LABEL_CLASS="M"
  LABEL_TAG="m"
  SEED="$(( slot - 5 ))"
fi

RUN_TAG="diag_mvit_${LABEL_TAG}w${WINDOW_HOUR}_t8_window_augprobe"

EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-6}"
HEAD_LR="${HEAD_LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
MVIT_MODEL_ID="${MVIT_MODEL_ID:-mvit_v1_b}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"

cmd=(python -m classifier_NN.legacy.run_experiments_A3_window
  --backbone mvit
  --window-hours "$WINDOW_HOUR"
  --seed "$SEED"
  --frame-count 8
  --seq-stride-steps 1
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --image-size 224
  --lr "$LR"
  --backbone-lr "$BACKBONE_LR"
  --head-lr "$HEAD_LR"
  --warmup-epochs "$WARMUP_EPOCHS"
  --grad-clip-norm "$GRAD_CLIP_NORM"
  --min-flare-class "$LABEL_CLASS"
  --loss-type weighted_bce
  --scheduler cosine
  --early-stopping-min-epoch 3
  --threshold-min-precision 0.15
  --threshold-min-recall 0.5
  --num-workers "$NUM_WORKERS"
  --use-multi-gpu
  --multi-gpu-max-devices "$NUM_GPUS"
  --pretrained-3d
  --pretrained-mvit
  --mvit-model-id "$MVIT_MODEL_ID"
  --balance-classes
  --auto-set-neg-keep-probs
  --target-neg-none 100000
  --neg-keep-prob-c 1.0
  --no-balanced-batch-sampling
  --use-aug
  --run-tag "$RUN_TAG"
)

echo "=========================================="
echo "Task ${idx}: model=mvit class=${LABEL_CLASS}+ window=${WINDOW_HOUR}h T=8 seed=${SEED} aug=1"
echo "  run_tag=${RUN_TAG}"
echo "  global_batch=${BATCH_SIZE} workers=${NUM_WORKERS} lr=${LR} backbone_lr=${BACKBONE_LR} head_lr=${HEAD_LR} warmup=${WARMUP_EPOCHS} clip=${GRAD_CLIP_NORM}"
echo "=========================================="
echo "Running: ${cmd[*]}"
"${cmd[@]}"
