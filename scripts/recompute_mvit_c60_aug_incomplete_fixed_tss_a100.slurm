#!/usr/bin/env bash
# Re-evaluate the three unfinished A3 60-minute augmentation runs on the test set
# using their validation-selected thresholds from the saved best_tss checkpoints.

#SBATCH --job-name=arfl-mvit-c60-aug-fix
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
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

export OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

EXP_DIRS=(
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:48:42_A3-16-cadence-60min_mvit_seed0_cplus_aug_dsnone100000_224_diag_mvit_c60_augprobe"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:54:45_A3-16-cadence-60min_mvit_seed3_cplus_aug_dsnone100000_224_diag_mvit_c60_augprobe"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:59:17_A3-16-cadence-60min_mvit_seed1_mplus_aug_dsnone100000_224_diag_mvit_m60_augprobe"
)
FIXED_THRESHOLDS=(
  0.216
  0.620
  0.0010023012117976577
)
SUFFIXES=(
  "fixedthr_testeval_mvit_c60_seed0_aug"
  "fixedthr_testeval_mvit_c60_seed3_aug"
  "fixedthr_testeval_mvit_m60_seed1_aug"
)

idx="${SLURM_ARRAY_TASK_ID}"
EXP_DIR="${EXP_DIRS[$idx]}"
FIXED_THRESHOLD="${FIXED_THRESHOLDS[$idx]}"
SUFFIX="${SUFFIXES[$idx]}"

cmd=(python -m classifier_NN.reevaluate_checkpoint
  "$EXP_DIR"
  --checkpoint best_tss.pt
  --suffix "$SUFFIX"
  --fixed-threshold "$FIXED_THRESHOLD"
)

echo "=========================================="
echo "Recompute fixed-threshold test metrics for incomplete A3 MViT c60+aug runs"
echo "=========================================="
echo "task=$idx"
echo "exp_dir=$EXP_DIR"
echo "fixed_threshold=$FIXED_THRESHOLD"
echo "suffix=$SUFFIX"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
