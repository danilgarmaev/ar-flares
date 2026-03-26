#!/usr/bin/env bash
#SBATCH --job-name=arfl-vswin-ddp-bm
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

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_GPUS="${NUM_GPUS:-2}"
IMAGE_SIZE=224
CADENCE="${CADENCE:-72}"
LABEL_CLASS="${LABEL_CLASS:-M}"
REGIME="${REGIME:-1}"
LOSS_TYPE="${LOSS_TYPE:-weighted_bce}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-512}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-64}"

if [[ "$NUM_GPUS" -lt 2 ]]; then
  echo "DDP benchmark expects NUM_GPUS>=2"
  exit 1
fi

echo "Benchmark mode: DDP"
echo "Config: class=${LABEL_CLASS}+ cadence=${CADENCE}min regime=${REGIME} loss=${LOSS_TYPE} epochs=${EPOCHS} batch_per_gpu=${BATCH_SIZE_PER_GPU} global_batch=$((BATCH_SIZE_PER_GPU * NUM_GPUS)) gpus=${NUM_GPUS} workers=${NUM_WORKERS} steps=${STEPS_PER_EPOCH} val_batches=${VAL_MAX_BATCHES}"

base_cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence
  --backbone videoswin
  --interval-min "$CADENCE"
  --seed "$SEED"
  --single
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE_PER_GPU"
  --image-size "$IMAGE_SIZE"
  --lr "$LR"
  --min-flare-class "$LABEL_CLASS"
  --loss-type "$LOSS_TYPE"
  --early-stopping-min-epoch 1
  --threshold-min-precision 0.15
  --threshold-min-recall 0.5
  --num-workers "$NUM_WORKERS"
  --steps-per-epoch "$STEPS_PER_EPOCH"
  --val-max-batches "$VAL_MAX_BATCHES"
)

if [[ "$REGIME" == "0" ]]; then
  base_cmd+=(--use-aug --no-balance-classes --no-balanced-batch-sampling)
else
  base_cmd+=(--no-aug --balance-classes --auto-set-neg-keep-probs --target-neg-none 100000 --neg-keep-prob-c 1.0 --no-balanced-batch-sampling)
fi

ddp_cmd=(torchrun --standalone --nproc_per_node "$NUM_GPUS" -m classifier_NN.legacy.run_experiments_16frames_cadence)
ddp_cmd+=("${base_cmd[@]:3}" --use-ddp --multi-gpu-max-devices "$NUM_GPUS")

echo "Running DDP benchmark: ${ddp_cmd[*]}"
time -p "${ddp_cmd[@]}"
