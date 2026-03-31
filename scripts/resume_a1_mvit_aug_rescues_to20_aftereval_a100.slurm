#!/usr/bin/env bash
# Warm-start the A1 MViT rescue runs again from their current resumed checkpoints to 20 epochs.

#SBATCH --job-name=arfl-a1mvit-re2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
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
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:59:50_A1-56_mvitv2_small_seed1_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:13:40_A1-112_mvitv2_small_seed1_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:14:21_A1-112_mvitv2_small_seed3_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:14:41_A1-112_mvitv2_small_seed4_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:18:23_A1-56_mvitv2_small_seed4_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
)
WARM_STARTS=(
  "resume_to15_mvit_cplus_56_seed1_aug.pt"
  "resume_to17_mvit_mplus_112_seed1_aug.pt"
  "resume_to14_mvit_mplus_112_seed3_aug.pt"
  "resume_to16_mvit_mplus_112_seed4_aug.pt"
  "resume_to17_mvit_mplus_56_seed4_aug.pt"
)
SUFFIXES=(
  "resume_to20_aftereval_mvit_cplus_56_seed1_aug"
  "resume_to20_aftereval_mvit_mplus_112_seed1_aug"
  "resume_to20_aftereval_mvit_mplus_112_seed3_aug"
  "resume_to20_aftereval_mvit_mplus_112_seed4_aug"
  "resume_to20_aftereval_mvit_mplus_56_seed4_aug"
)

idx="${SLURM_ARRAY_TASK_ID}"
EXP_DIR="${EXP_DIRS[$idx]}"
WARM_START="${WARM_STARTS[$idx]}"
SUFFIX="${SUFFIXES[$idx]}"

cmd=(python -m classifier_NN.resume_experiment
  "$EXP_DIR"
  --warm-start "$WARM_START"
  --baseline-best-tss "$WARM_START"
  --resume-full-ckpt last_full.pt
  --max-epochs 20
  --suffix "$SUFFIX"
)

echo "=========================================="
echo "Second A1 rescue warm-start continuation"
echo "=========================================="
echo "task=$idx"
echo "exp_dir=$EXP_DIR"
echo "warm_start=$WARM_START"
echo "suffix=$SUFFIX"
echo "Running: ${cmd[*]}"
"${cmd[@]}"

