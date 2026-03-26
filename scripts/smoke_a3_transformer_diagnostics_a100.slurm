#!/usr/bin/env bash
#SBATCH --job-name=arfl-a3diag-smk
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

MODEL="${MODEL:-mvit}"
MVIT_MODEL_ID="${MVIT_MODEL_ID:-mvit_v1_b}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"

cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence
  --backbone "$MODEL"
  --interval-min 108
  --seed 0
  --single
  --epochs 4
  --batch-size "$BATCH_SIZE"
  --image-size 224
  --lr 1e-4
  --backbone-lr 1e-6
  --head-lr 1e-4
  --warmup-epochs 1
  --grad-clip-norm 1.0
  --min-flare-class C
  --loss-type weighted_bce
  --scheduler cosine
  --num-workers "$NUM_WORKERS"
  --steps-per-epoch 64
  --val-max-batches 16
  --early-stopping-min-epoch 1
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
  --pretrained-3d
  --pretrained-mvit
  --mvit-model-id "$MVIT_MODEL_ID"
  --run-tag smoke_diag_mvit_c108
)

echo "Running diagnostic smoke: ${cmd[*]}"
"${cmd[@]}"
