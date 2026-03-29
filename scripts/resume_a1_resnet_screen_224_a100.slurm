#!/usr/bin/env bash
# Resume the two interrupted A1 ResNet-50 screen2d224 runs and write fixed-threshold metrics.
# Usage:
#   sbatch scripts/resume_a1_resnet_screen_224_a100.slurm
#
# Array mapping:
#   0 -> C+  results/2026-03-27 20:46:14_A1-224_resnet50_seed0_screen2d224_cplus_dsnone300k_wbce_ftv2
#   1 -> M+  results/2026-03-27 21:23:39_A1-224_resnet50_seed0_screen2d224_mplus_dsnone100k_wbce_ftv2

#SBATCH --job-name=arfl-a1resnet-resume
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1

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
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-27 20:46:14_A1-224_resnet50_seed0_screen2d224_cplus_dsnone300k_wbce_ftv2"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-27 21:23:39_A1-224_resnet50_seed0_screen2d224_mplus_dsnone100k_wbce_ftv2"
)
SUFFIXES=(
  "resume_to40_resnet_cplus_ftv2"
  "resume_to40_resnet_mplus_ftv2"
)

idx="${SLURM_ARRAY_TASK_ID}"
EXP_DIR="${EXP_DIRS[$idx]}"
SUFFIX="${SUFFIXES[$idx]}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"

cmd=(python -m classifier_NN.resume_experiment
  "$EXP_DIR"
  --warm-start best_tss.pt
  --baseline-best-tss best_tss.pt
  --resume-full-ckpt last_full.pt
  --max-epochs "$MAX_EPOCHS"
  --suffix "$SUFFIX"
)

echo "=========================================="
echo "Resume A1 ResNet screen2d224 run"
echo "=========================================="
echo "task=$idx"
echo "exp_dir=$EXP_DIR"
echo "max_epochs=$MAX_EPOCHS suffix=$SUFFIX"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
