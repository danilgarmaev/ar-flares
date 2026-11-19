#!/usr/bin/env bash
# Longer single-frame MViT LearnableScaler smoke with identity-initialized scaler.
# Purpose: check whether identity init fixes the immediate A1 MViT NaNs.
#SBATCH --job-name=arfl-a1-mvit-lsc-smk
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

source /home/dgarmaev/envs/ar-flares/bin/activate

export AR_FLARES_WDS_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_flow
export AR_FLARES_RESULTS_BASE=/home/dgarmaev/scratch/ar-flares/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/home/dgarmaev/scratch/ar-flares/data

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID:-a1-mvit-lsc-smk}
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

LR_VALUES=(${LR_VALUES:-1e-4 5e-5 2e-5})
LR="${LR_VALUES[$SLURM_ARRAY_TASK_ID]}"

MVIT_MODEL_ID="${MVIT_MODEL_ID:-mvit_v1_b}"
SCALER_LR="${SCALER_LR:-1e-3}"
SEED="${SEED:-0}"

cmd=(python -m classifier_NN.legacy.run_experiments_A1
  --single
  --backbone mvit
  --mvit-model-id "$MVIT_MODEL_ID"
  --mvit-remove-temporal-conv
  --resolution 224
  --seed "$SEED"
  --epochs "${EPOCHS:-2}"
  --batch-size "${BATCH_SIZE:-16}"
  --lr "$LR"
  --optimizer adamw
  --weight-decay "${WEIGHT_DECAY:-0.05}"
  --scheduler "${SCHEDULER:-none}"
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}"
  --steps-per-epoch "${STEPS_PER_EPOCH:-1200}"
  --max-train-shards "${MAX_TRAIN_SHARDS:-32}"
  --max-val-shards "${MAX_VAL_SHARDS:-8}"
  --max-test-shards "${MAX_TEST_SHARDS:-8}"
  --no-freeze-backbone
  --no-pretrained
  --use-aug
  --aug-preset "${AUG_PRESET:-robust}"
  --learnable-scaler
  --learnable-scaler-module learnable_scaler
  --learnable-scaler-class LearnableScaler
  --learnable-scaler-lr "$SCALER_LR"
  --name-suffix "mvitv1_notemporal_lsc_idinit_longsmoke_lr${LR}")

echo "Running MViT LearnableScaler long smoke: ${cmd[*]}"
"${cmd[@]}"
