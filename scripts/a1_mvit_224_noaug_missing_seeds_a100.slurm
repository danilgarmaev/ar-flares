#!/usr/bin/env bash
# Fill in the missing no-augmentation seeds for A1 2D MViTv2-Small at 224 resolution.
# Existing seed 0 runs already exist for both C+ and M+; this script runs seeds 1..4.
# Usage:
#   sbatch scripts/a1_mvit_224_noaug_missing_seeds_a100.slurm

#SBATCH --job-name=arfl-a1-mvit224-noa
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-7

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

LABEL_CLASSES=(C C C C M M M M)
SEEDS=(1 2 3 4 1 2 3 4)

idx="${SLURM_ARRAY_TASK_ID}"
LABEL_CLASS="${LABEL_CLASSES[$idx]}"
SEED="${SEEDS[$idx]}"
RESOLUTION=224

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-6}"
HEAD_LR="${HEAD_LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-cosine}"
LOSS_TYPE="${LOSS_TYPE:-weighted_bce}"
PRETRAINED="${PRETRAINED:-1}"
BALANCE_CLASSES="${BALANCE_CLASSES:-1}"
BALANCE_MODE="${BALANCE_MODE:-prob}"
AUTO_SET_NEG_KEEP_PROBS="${AUTO_SET_NEG_KEEP_PROBS:-1}"

if [[ "$LABEL_CLASS" == "C" ]]; then
  TARGET_NEG_NONE="${TARGET_NEG_NONE_C:-300000}"
  NEG_KEEP_PROB_C="${NEG_KEEP_PROB_C_C:-}"
  NAME_SUFFIX="${NAME_SUFFIX_C:-screen2d224_cplus_dsnone300k_wbce_ftv2}"
else
  TARGET_NEG_NONE="${TARGET_NEG_NONE_M:-100000}"
  NEG_KEEP_PROB_C="${NEG_KEEP_PROB_C_M:-1.0}"
  NAME_SUFFIX="${NAME_SUFFIX_M:-screen2d224_mplus_dsnone100k_wbce_ftv2}"
fi

cmd=(python -m classifier_NN.legacy.run_experiments_A1
  --single
  --backbone mvitv2_small
  --resolution "$RESOLUTION"
  --seed "$SEED"
  --name-suffix "$NAME_SUFFIX"
  --min-flare-class "$LABEL_CLASS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --optimizer "$OPTIMIZER"
  --scheduler "$SCHEDULER"
  --loss-type "$LOSS_TYPE"
  --backbone-lr "$BACKBONE_LR"
  --head-lr "$HEAD_LR"
  --warmup-epochs "$WARMUP_EPOCHS"
  --grad-clip-norm "$GRAD_CLIP_NORM"
  --no-freeze-backbone
  --no-aug)

if [[ "$BALANCE_CLASSES" == "1" ]]; then
  cmd+=(--balance-classes)
else
  cmd+=(--no-balance-classes)
fi
if [[ -n "$BALANCE_MODE" ]]; then
  cmd+=(--balance-mode "$BALANCE_MODE")
fi
if [[ "$AUTO_SET_NEG_KEEP_PROBS" == "1" ]]; then
  cmd+=(--auto-set-neg-keep-probs)
else
  cmd+=(--no-auto-set-neg-keep-probs)
fi
if [[ -n "$TARGET_NEG_NONE" ]]; then
  cmd+=(--target-neg-none "$TARGET_NEG_NONE")
fi
if [[ -n "$NEG_KEEP_PROB_C" ]]; then
  cmd+=(--neg-keep-prob-c "$NEG_KEEP_PROB_C")
fi
if [[ "$PRETRAINED" == "1" ]]; then
  cmd+=(--pretrained)
else
  cmd+=(--no-pretrained)
fi

echo "=========================================="
echo "Task ${idx}: model=mvitv2_small class=${LABEL_CLASS}+ res=${RESOLUTION} seed=${SEED} aug=0"
echo "  batch=${BATCH_SIZE} lr=${LR} backbone_lr=${BACKBONE_LR} head_lr=${HEAD_LR} warmup=${WARMUP_EPOCHS} clip=${GRAD_CLIP_NORM}"
echo "=========================================="
echo "Running: ${cmd[*]}"
"${cmd[@]}"
