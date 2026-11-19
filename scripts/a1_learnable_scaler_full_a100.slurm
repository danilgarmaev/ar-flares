#!/usr/bin/env bash
# Full from-scratch single-frame run for Anh Tuan's LearnableScaler normalization.
# Submit after smoke LRs confirm no NaN/overflow.
#SBATCH --job-name=arfl-a1-lsc-full
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID:-a1-lsc-full}
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

BACKBONE="${BACKBONE:-mvit}"
RESOLUTION="${RESOLUTION:-224}"
SEED="${SEED:-0}"
LR="${LR:-5e-5}"
SCALER_LR="${SCALER_LR:-1e-3}"
LEARNABLE_SCALER_MODULE="${LEARNABLE_SCALER_MODULE:-learnable_scaler}"
LEARNABLE_SCALER_CLASS="${LEARNABLE_SCALER_CLASS:-LearnableScaler}"

cmd=(python -m classifier_NN.legacy.run_experiments_A1
  --single
  --backbone "$BACKBONE"
  --resolution "$RESOLUTION"
  --seed "$SEED"
  --epochs "${EPOCHS:-50}"
  --batch-size "${BATCH_SIZE:-64}"
  --lr "$LR"
  --optimizer adamw
  --weight-decay "${WEIGHT_DECAY:-0.05}"
  --scheduler "${SCHEDULER:-cosine}"
  --grad-clip-norm "${GRAD_CLIP_NORM:-1.0}"
  --no-freeze-backbone
  --no-pretrained
  --use-aug
  --aug-preset "${AUG_PRESET:-robust}"
  --learnable-scaler
  --learnable-scaler-module "$LEARNABLE_SCALER_MODULE"
  --learnable-scaler-class "$LEARNABLE_SCALER_CLASS"
  --learnable-scaler-lr "$SCALER_LR"
  --name-suffix "lsc_adamw_scratch_lr${LR}")

if [[ "$BACKBONE" == "mvit" || "$BACKBONE" == mvit_* ]]; then
  cmd+=(--mvit-model-id "${MVIT_MODEL_ID:-mvit_v1_b}" --mvit-remove-temporal-conv)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
