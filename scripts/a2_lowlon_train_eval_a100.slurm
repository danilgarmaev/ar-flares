#!/usr/bin/env bash
#SBATCH --job-name=arfl-a2lon-train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

# Matches what worked in your other project; harmless if unused.
module load gcc opencv

# Project venv
source /home/dgarmaev/envs/ar-flares/bin/activate

# Force dataset + results locations (do not rely on heuristics)
export AR_FLARES_WDS_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_flow
export AR_FLARES_RESULTS_BASE=/home/dgarmaev/scratch/ar-flares/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/home/dgarmaev/scratch/ar-flares/data

# Offline safety switches: fail fast instead of hanging on network retries.
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Put caches on scratch so they can be reused across jobs.
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ---------------------------
# A2 low-longitude train+eval
# ---------------------------
# Required:
#   EXP_DIR: baseline experiment directory (must contain config.json and metrics.json)
# Optional:
#   CHECKPOINT: checkpoint filename inside EXP_DIR (default: best_tss.pt)
#   SRS_ROOT: path to SRS text files
#   TRAIN_LON_ABS_MAX: default 30
#   TRAIN_SEED: seed for the low-lon training run
#
# Example:
#   sbatch --export=ALL,EXP_DIR="/path/to/best_run" scripts/a2_lowlon_train_eval_a100.slurm

EXP_DIR="${EXP_DIR:-}"
CHECKPOINT="${CHECKPOINT:-best_tss.pt}"
SRS_ROOT="${SRS_ROOT:-/home/dgarmaev/scratch/ar-flares/data/SRS}"
TRAIN_LON_ABS_MAX="${TRAIN_LON_ABS_MAX:-30}"
TRAIN_SEED="${TRAIN_SEED:-0}"

if [[ -z "$EXP_DIR" ]]; then
  echo "ERROR: Provide EXP_DIR via sbatch --export=ALL,EXP_DIR=..." >&2
  exit 2
fi

cmd=(python -m classifier_NN.legacy.run_experiments_A2 \
  --exp-dir "$EXP_DIR" \
  --checkpoint "$CHECKPOINT" \
  --srs-root "$SRS_ROOT" \
  --train-lon-abs-max "$TRAIN_LON_ABS_MAX" \
  --train-seed "$TRAIN_SEED")

echo "Running: ${cmd[*]}"
"${cmd[@]}"
