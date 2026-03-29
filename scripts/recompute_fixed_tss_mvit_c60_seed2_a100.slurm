#!/usr/bin/env bash
# Re-evaluate the resumed A3 MViT 60-minute C+ seed=2 checkpoint at the
# validation-selected threshold so fixed_TSS is available.
#
# Usage:
#   sbatch scripts/recompute_fixed_tss_mvit_c60_seed2_a100.slurm

#SBATCH --job-name=arfl-mvit-c60-fixthr
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
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

EXP_DIR="${EXP_DIR:-/home/dgarmaev/scratch/ar-flares/results/2026-03-27 00:07:39_A3-16-cadence-60min_mvit_seed2_cplus_noaug_dsnone100000_224_diag_mvit_c60}"
CHECKPOINT="${CHECKPOINT:-resume_to20_seed2_c60.pt}"
SUFFIX="${SUFFIX:-fixedthr_resume_to20_seed2_c60}"
FIXED_THRESHOLD="${FIXED_THRESHOLD:-0.9989663584937807}"

cmd=(python -m classifier_NN.reevaluate_checkpoint
  "$EXP_DIR"
  --checkpoint "$CHECKPOINT"
  --suffix "$SUFFIX"
  --fixed-threshold "$FIXED_THRESHOLD"
)

echo "=========================================="
echo "Recompute fixed-threshold metrics for resumed MViT c60 seed2"
echo "=========================================="
echo "exp_dir=$EXP_DIR"
echo "checkpoint=$CHECKPOINT"
echo "fixed_threshold=$FIXED_THRESHOLD"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
