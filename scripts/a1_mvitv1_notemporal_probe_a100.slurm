#!/usr/bin/env bash
# Single-setting A1 probe: MViT v1 on single-frame inputs with the temporal
# patch embedding removed, to verify the A1 retraining path before launching a
# larger sweep.
#
# Usage:
#   sbatch scripts/a1_mvitv1_notemporal_probe_a100.slurm

#SBATCH --job-name=arfl-a1-mvitv1-probe
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
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

RESOLUTION="${RESOLUTION:-224}"
SEED="${SEED:-0}"
LABEL_CLASS="${LABEL_CLASS:-C}"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-6}"
HEAD_LR="${HEAD_LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-cosine}"
LOSS_TYPE="${LOSS_TYPE:-weighted_bce}"
MAX_TRAIN_SHARDS="${MAX_TRAIN_SHARDS:-8}"
MAX_VAL_SHARDS="${MAX_VAL_SHARDS:-2}"
MAX_TEST_SHARDS="${MAX_TEST_SHARDS:-2}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
TARGET_NEG_NONE="${TARGET_NEG_NONE:-300000}"
NEG_KEEP_PROB_C="${NEG_KEEP_PROB_C:-}"
NAME_SUFFIX="${NAME_SUFFIX:-screen2d224_mvitv1_notemporal_probe}"

cmd=(python -m classifier_NN.legacy.run_experiments_A1
  --single
  --backbone mvit
  --mvit-model-id mvit_v1_b
  --mvit-remove-temporal-conv
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
  --max-train-shards "$MAX_TRAIN_SHARDS"
  --max-val-shards "$MAX_VAL_SHARDS"
  --max-test-shards "$MAX_TEST_SHARDS"
  --steps-per-epoch "$STEPS_PER_EPOCH"
  --no-freeze-backbone
  --use-aug
  --balance-classes
  --auto-set-neg-keep-probs
  --target-neg-none "$TARGET_NEG_NONE"
  --pretrained)

if [[ -n "$NEG_KEEP_PROB_C" ]]; then
  cmd+=(--neg-keep-prob-c "$NEG_KEEP_PROB_C")
fi

echo "=========================================="
echo "A1 single-frame MViT v1 probe"
echo "  label=${LABEL_CLASS}+ res=${RESOLUTION} seed=${SEED}"
echo "  epochs=${EPOCHS} batch=${BATCH_SIZE} steps=${STEPS_PER_EPOCH}"
echo "  max_train_shards=${MAX_TRAIN_SHARDS} max_val_shards=${MAX_VAL_SHARDS} max_test_shards=${MAX_TEST_SHARDS}"
echo "=========================================="
echo "Running: ${cmd[*]}"
"${cmd[@]}"
