#!/usr/bin/env bash
# Retry the two stuck 300-minute C+ augmented MViT runs with fewer dataloader workers.
# Usage:
#   sbatch scripts/run_mvit_cadence_aug_300_retry_cplus_seed0_3_a100.slurm

#SBATCH --job-name=arfl-mvit-c300-re
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1

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

SEEDS=(0 3)
idx="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$idx]}"

CADENCE="${CADENCE:-300}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-6}"
HEAD_LR="${HEAD_LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
MVIT_MODEL_ID="${MVIT_MODEL_ID:-mvit_v1_b}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"

cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence
  --backbone mvit
  --interval-min "$CADENCE"
  --seed "$SEED"
  --single
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --image-size 224
  --lr "$LR"
  --backbone-lr "$BACKBONE_LR"
  --head-lr "$HEAD_LR"
  --warmup-epochs "$WARMUP_EPOCHS"
  --grad-clip-norm "$GRAD_CLIP_NORM"
  --min-flare-class C
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
  --run-tag diag_mvit_c300_augprobe_retry
)

echo "=========================================="
echo "Retry task ${idx}: model=mvit class=C+ cadence=${CADENCE} seed=${SEED} aug=1 workers=${NUM_WORKERS}"
echo "=========================================="
echo "Running: ${cmd[*]}"
"${cmd[@]}"
