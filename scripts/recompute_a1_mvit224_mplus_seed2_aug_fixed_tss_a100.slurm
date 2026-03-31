#!/usr/bin/env bash
# Re-evaluate the interrupted A1 224 M+ seed=2 MViT augmentation run on Test only
# using its validation-selected threshold.

#SBATCH --job-name=arfl-a1mvit-reeval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

EXP_DIR="${EXP_DIR:-/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:05:36_A1-224_mvitv2_small_seed2_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug}"
CHECKPOINT="${CHECKPOINT:-best_tss.pt}"
FIXED_THRESHOLD="${FIXED_THRESHOLD:-0.006986619509252208}"
SUFFIX="${SUFFIX:-fixedthr_testeval_mvit_mplus_224_seed2_aug}"

cmd=(python -m classifier_NN.reevaluate_checkpoint
  "$EXP_DIR"
  --checkpoint "$CHECKPOINT"
  --suffix "$SUFFIX"
  --fixed-threshold "$FIXED_THRESHOLD"
)

echo "=========================================="
echo "Recompute fixed-threshold test metrics"
echo "=========================================="
echo "exp_dir=$EXP_DIR"
echo "checkpoint=$CHECKPOINT"
echo "fixed_threshold=$FIXED_THRESHOLD"
echo "suffix=$SUFFIX"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
