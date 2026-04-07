#!/usr/bin/env bash
# 5-seed A3 frame-count study on the unique label/cadence settings:
#   C+ @ 300 min, C+ @ 180 min, M+ @ 180 min

#SBATCH --job-name=arfl-mvit-fcnt
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-29

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

idx="${SLURM_ARRAY_TASK_ID}"
group=$(( idx / 10 ))
slot=$(( idx % 10 ))

if (( group == 0 )); then
  LABEL_CLASS="C"
  CADENCE="300"
  PREFIX="c300"
  OFFSETS_T8="-375,-325,-275,-225,-150,-100,-50,0"
  OFFSETS_T4="-375,-250,-125,0"
elif (( group == 1 )); then
  LABEL_CLASS="C"
  CADENCE="180"
  PREFIX="c180"
  OFFSETS_T8="-225,-195,-165,-135,-90,-60,-30,0"
  OFFSETS_T4="-225,-150,-75,0"
else
  LABEL_CLASS="M"
  CADENCE="180"
  PREFIX="m180"
  OFFSETS_T8="-225,-195,-165,-135,-90,-60,-30,0"
  OFFSETS_T4="-225,-150,-75,0"
fi

if (( slot < 5 )); then
  T_FRAMES="8"
  SEED="${slot}"
  SEQ_OFFSETS="${OFFSETS_T8}"
else
  T_FRAMES="4"
  SEED="$(( slot - 5 ))"
  SEQ_OFFSETS="${OFFSETS_T4}"
fi

RUN_TAG="diag_mvit_${PREFIX}_t${T_FRAMES}_framecount_augprobe"

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

cmd=(python -m classifier_NN.legacy.run_experiments_A3_framecount
  --backbone mvit
  --cadence-min "$CADENCE"
  --seed "$SEED"
  --frame-count "$T_FRAMES"
  "--seq-offsets=$SEQ_OFFSETS"
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
echo "Task ${idx}: class=${LABEL_CLASS}+ cadence=${CADENCE} T=${T_FRAMES} seed=${SEED}"
echo "  offsets=${SEQ_OFFSETS}"
echo "  run_tag=${RUN_TAG}"
echo "=========================================="
echo "Running: ${cmd[*]}"
"${cmd[@]}"
