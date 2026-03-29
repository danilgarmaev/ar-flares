#!/usr/bin/env bash
# Resume the interrupted A3 MViT 60-minute cadence C+ seed=2 run to 20 epochs.
# Usage:
#   sbatch scripts/resume_mvit_c60_seed2_a100.slurm
# Optional overrides:
#   sbatch --export=ALL,MAX_EPOCHS=22,SUFFIX=resume_to22_seed2_c60 scripts/resume_mvit_c60_seed2_a100.slurm

#SBATCH --job-name=arfl-resume-mvit-c60-s2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
MAX_EPOCHS="${MAX_EPOCHS:-20}"
SUFFIX="${SUFFIX:-resume_to20_seed2_c60}"
WARM_START="${WARM_START:-best_tss.pt}"
BASELINE_BEST_TSS="${BASELINE_BEST_TSS:-best_tss.pt}"
RESUME_FULL_CKPT="${RESUME_FULL_CKPT:-last_full.pt}"

cmd=(python -m classifier_NN.resume_experiment
  "$EXP_DIR"
  --warm-start "$WARM_START"
  --baseline-best-tss "$BASELINE_BEST_TSS"
  --resume-full-ckpt "$RESUME_FULL_CKPT"
  --max-epochs "$MAX_EPOCHS"
  --suffix "$SUFFIX"
)

echo "=========================================="
echo "Resume A3 MViT cadence run"
echo "=========================================="
echo "exp_dir=$EXP_DIR"
echo "max_epochs=$MAX_EPOCHS suffix=$SUFFIX"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
