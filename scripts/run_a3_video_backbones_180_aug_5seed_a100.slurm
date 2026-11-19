#!/usr/bin/env bash
# 5-seed augmented 180-minute video-backbone comparison.
#
# Models:
#   - VideoMAE, Video Swin at 224x224
#   - R(2+1)D-18 at 112x112, matching the earlier R(2+1)D protocol
#
# Tasks:
#   3 backbones x 2 flare thresholds x 5 seeds = 30 array tasks

#SBATCH --job-name=arfl-a3vid180aug
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-29

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

BACKBONES=(
  videomae videomae videomae videomae videomae
  videomae videomae videomae videomae videomae
  videoswin videoswin videoswin videoswin videoswin
  videoswin videoswin videoswin videoswin videoswin
  r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18
  r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18 r2plus1d_18
)
LABEL_CLASSES=(
  C C C C C
  M M M M M
  C C C C C
  M M M M M
  C C C C C
  M M M M M
)
SEEDS=(
  0 1 2 3 4
  0 1 2 3 4
  0 1 2 3 4
  0 1 2 3 4
  0 1 2 3 4
  0 1 2 3 4
)
RUN_TAGS=(
  diag_vmae_c180_aug diag_vmae_c180_aug diag_vmae_c180_aug diag_vmae_c180_aug diag_vmae_c180_aug
  diag_vmae_m180_aug diag_vmae_m180_aug diag_vmae_m180_aug diag_vmae_m180_aug diag_vmae_m180_aug
  diag_vswin_c180_aug diag_vswin_c180_aug diag_vswin_c180_aug diag_vswin_c180_aug diag_vswin_c180_aug
  diag_vswin_m180_aug diag_vswin_m180_aug diag_vswin_m180_aug diag_vswin_m180_aug diag_vswin_m180_aug
  diag_r2p1d_c180_aug diag_r2p1d_c180_aug diag_r2p1d_c180_aug diag_r2p1d_c180_aug diag_r2p1d_c180_aug
  diag_r2p1d_m180_aug diag_r2p1d_m180_aug diag_r2p1d_m180_aug diag_r2p1d_m180_aug diag_r2p1d_m180_aug
)

idx="${SLURM_ARRAY_TASK_ID}"
MODEL="${BACKBONES[$idx]}"
LABEL_CLASS="${LABEL_CLASSES[$idx]}"
SEED="${SEEDS[$idx]}"
RUN_TAG="${RUN_TAGS[$idx]}"

CADENCE="${CADENCE:-180}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
BACKBONE_LR="${BACKBONE_LR:-1e-6}"
HEAD_LR="${HEAD_LR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"

IMAGE_SIZE=224
if [[ "$MODEL" == "r2plus1d_18" ]]; then
  IMAGE_SIZE=112
fi

cmd=(python -m classifier_NN.legacy.run_experiments_16frames_cadence
  --backbone "$MODEL"
  --interval-min "$CADENCE"
  --seed "$SEED"
  --single
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --image-size "$IMAGE_SIZE"
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
  --balance-classes
  --auto-set-neg-keep-probs
  --target-neg-none 100000
  --neg-keep-prob-c 1.0
  --no-balanced-batch-sampling
  --use-aug
  --run-tag "$RUN_TAG"
)

echo "=========================================="
echo "Task ${idx}: model=${MODEL} class=${LABEL_CLASS}+ cadence=${CADENCE} seed=${SEED} aug=1 run_tag=${RUN_TAG}"
echo "  image_size=${IMAGE_SIZE} global_batch=${BATCH_SIZE} workers=${NUM_WORKERS} lr=${LR}"
echo "  backbone_lr=${BACKBONE_LR} head_lr=${HEAD_LR} warmup=${WARMUP_EPOCHS} clip=${GRAD_CLIP_NORM}"
echo "=========================================="
echo "Running: ${cmd[*]}"
"${cmd[@]}"
