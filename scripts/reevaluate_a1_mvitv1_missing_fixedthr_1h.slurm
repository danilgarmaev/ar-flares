#!/usr/bin/env bash
# Re-evaluate incomplete A1 MViTv1 runs on the Test split only, using the
# validation-selected threshold from the best validation-TSS epoch.
#
# Usage:
#   sbatch scripts/reevaluate_a1_mvitv1_missing_fixedthr_1h.slurm

#SBATCH --job-name=arfl-a1v1-reeval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4

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
  "/home/dgarmaev/scratch/ar-flares/results/2026-04-27 19:26:57_A1-112_mvit_seed2_screen2dres_mvitv1_notemporal_cplus_dsnone300k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-04-27 19:32:08_A1-56_mvit_seed2_screen2dres_mvitv1_notemporal_cplus_dsnone300k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-04-27 19:37:49_A1-224_mvit_seed1_screen2dres_mvitv1_notemporal_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-04-27 19:42:45_A1-56_mvit_seed0_screen2dres_mvitv1_notemporal_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-04-27 19:42:51_A1-56_mvit_seed3_screen2dres_mvitv1_notemporal_mplus_dsnone100k_wbce_ftv2_aug"
)

FIXED_THRESHOLDS=(
  "0.4753504085616442"
  "0.5690000000000001"
  "0.011"
  "0.05391111595695838"
  "0.025"
)

SUFFIXES=(
  "fixedthr_testeval_mvitv1_cplus_112_seed2_aug"
  "fixedthr_testeval_mvitv1_cplus_56_seed2_aug"
  "fixedthr_testeval_mvitv1_mplus_224_seed1_aug"
  "fixedthr_testeval_mvitv1_mplus_56_seed0_aug"
  "fixedthr_testeval_mvitv1_mplus_56_seed3_aug"
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
echo "A1 MViTv1 fixed-threshold test re-evaluation"
echo "=========================================="
echo "task=${idx}"
echo "exp_dir=${EXP_DIR}"
echo "fixed_threshold=${FIXED_THRESHOLD}"
echo "suffix=${SUFFIX}"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
