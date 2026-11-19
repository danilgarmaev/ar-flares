#!/usr/bin/env bash
# Fast LR smoke sweep for Anh Tuan's LearnableScaler normalization.
# Requires learnable_scaler.py to be importable from the project root unless
# LEARNABLE_SCALER_MODULE is overridden.
#SBATCH --job-name=arfl-a1-lsc-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID:-a1-lsc-smoke}
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

BACKBONE="${BACKBONE:-vit_small_patch16_224}"
RESOLUTION="${RESOLUTION:-224}"
SEED="${SEED:-0}"
SCALER_LR="${SCALER_LR:-1e-3}"
LEARNABLE_SCALER_MODULE="${LEARNABLE_SCALER_MODULE:-learnable_scaler}"
LEARNABLE_SCALER_CLASS="${LEARNABLE_SCALER_CLASS:-LearnableScaler}"
LR_VALUES=(${LR_VALUES:-8e-4 4e-4 2e-4})
LR="${LR_VALUES[$SLURM_ARRAY_TASK_ID]}"

cmd=(python -m classifier_NN.legacy.run_experiments_A1
  --single
  --backbone "$BACKBONE"
  --resolution "$RESOLUTION"
  --seed "$SEED"
  --epochs "${EPOCHS:-1}"
  --batch-size "${BATCH_SIZE:-32}"
  --lr "$LR"
  --optimizer adamw
  --weight-decay "${WEIGHT_DECAY:-0.05}"
  --scheduler "${SCHEDULER:-none}"
  --steps-per-epoch "${STEPS_PER_EPOCH:-25}"
  --max-train-shards "${MAX_TRAIN_SHARDS:-2}"
  --max-val-shards "${MAX_VAL_SHARDS:-1}"
  --max-test-shards "${MAX_TEST_SHARDS:-1}"
  --no-freeze-backbone
  --no-pretrained
  --no-aug
  --learnable-scaler
  --learnable-scaler-module "$LEARNABLE_SCALER_MODULE"
  --learnable-scaler-class "$LEARNABLE_SCALER_CLASS"
  --learnable-scaler-lr "$SCALER_LR"
  --name-suffix "lsc_adamw_scratch_smoke_lr${LR}")

if [[ "$BACKBONE" == "mvit" || "$BACKBONE" == mvit_* ]]; then
  cmd+=(--mvit-model-id "${MVIT_MODEL_ID:-mvit_v1_b}" --mvit-remove-temporal-conv)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
