#!/usr/bin/env bash
# Resume the three unfinished A3 60-minute augmentation runs to 30 total epochs.

#SBATCH --job-name=arfl-resume-c60-aug
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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
MAX_EPOCHS=(
  30
  30
  30
)
SUFFIXES=(
  "resume_to30_mvit_c60_seed0_aug"
  "resume_to30_mvit_c60_seed3_aug"
  "resume_to30_mvit_m60_seed1_aug"
)

idx="${SLURM_ARRAY_TASK_ID}"
EXP_DIR="${EXP_DIRS[$idx]}"
TARGET_EPOCHS="${MAX_EPOCHS[$idx]}"
SUFFIX="${SUFFIXES[$idx]}"

cmd=(python -m classifier_NN.resume_experiment
  "$EXP_DIR"
  --warm-start best_tss.pt
  --baseline-best-tss best_tss.pt
  --resume-full-ckpt last_full.pt
  --max-epochs "$TARGET_EPOCHS"
  --suffix "$SUFFIX"
)

echo "=========================================="
echo "Resume incomplete A3 MViT c60+aug runs"
echo "=========================================="
echo "task=$idx"
echo "exp_dir=$EXP_DIR"
echo "max_epochs=$TARGET_EPOCHS"
echo "suffix=$SUFFIX"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
