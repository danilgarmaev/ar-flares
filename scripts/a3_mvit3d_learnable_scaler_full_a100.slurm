#!/usr/bin/env bash
# Full true 3D MViT LearnableScaler run on 16-frame sequences.
# Runs from scratch: no pretrained MViT weights.
# Assumes learnable_scaler.py uses identity initialization (a=1, b=0).
#SBATCH --job-name=arfl-a3-mvit-lsc-full
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=24:00:00
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
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID:-a3-mvit-lsc-full}
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

LR_VALUES=(${LR_VALUES:-1e-4 5e-5})
LR="${LR_VALUES[$SLURM_ARRAY_TASK_ID]}"

MVIT_MODEL_ID="${MVIT_MODEL_ID:-mvit_v1_b}"
SCALER_LR="${SCALER_LR:-1e-3}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CADENCE="${CADENCE:-108}"
SEED="${SEED:-0}"

cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence
  --backbone mvit
  --interval-min "$CADENCE"
  --seed "$SEED"
  --single
  --epochs "${EPOCHS:-50}"
  --batch-size "$BATCH_SIZE"
  --image-size 224
  --lr "$LR"
  --optimizer adamw
  --weight-decay "${WEIGHT_DECAY:-0.05}"
  --scheduler "${SCHEDULER:-cosine}"
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}"
  --min-flare-class C
  --loss-type weighted_bce
  --num-workers "$NUM_WORKERS"
  --early-stopping-min-epoch "${EARLY_STOPPING_MIN_EPOCH:-3}"
  --threshold-min-precision 0.15
  --threshold-min-recall 0.5
  --balance-classes
  --auto-set-neg-keep-probs
  --target-neg-none 100000
  --neg-keep-prob-c 1.0
  --no-balanced-batch-sampling
  --no-aug
  --use-multi-gpu
  --multi-gpu-max-devices "$NUM_GPUS"
  --no-pretrained-mvit
  --mvit-model-id "$MVIT_MODEL_ID"
  --learnable-scaler
  --learnable-scaler-module learnable_scaler
  --learnable-scaler-class LearnableScaler
  --learnable-scaler-lr "$SCALER_LR"
  --run-tag "mvit3d_lsc_idinit_full_lr${LR}")

echo "Running full MViT3D LearnableScaler experiment: ${cmd[*]}"
"${cmd[@]}"
