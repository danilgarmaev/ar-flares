#!/usr/bin/env bash
#SBATCH --job-name=arfl-a2lon-eval
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
# A2 configuration
# ---------------------------
# Provide either EXP_DIR (must contain config.json) or CFG_JSON.
# Optionally provide CHECKPOINT_PATH to point directly to a .pt file.
#
# IMPORTANT:
#   If THRESHOLD is not set, the Python script will try to auto-detect
#   Best_threshold from EXP_DIR/metrics.json (or EXP_DIR/log.txt).
#
# Typical usage:
#   sbatch --export=ALL,EXP_DIR="/path/to/exp",CHECKPOINT="best_tss.pt" scripts/a2_longitude_eval_cpu.slurm

EXP_DIR="${EXP_DIR:-}"
CFG_JSON="${CFG_JSON:-}"
CHECKPOINT="${CHECKPOINT:-best_tss.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
SRS_ROOT="${SRS_ROOT:-/home/dgarmaev/scratch/ar-flares/data/SRS}"
THRESHOLD="${THRESHOLD:-}"
TRAIN_LON_ABS_MAX="${TRAIN_LON_ABS_MAX:-}"
TRAIN_SEED="${TRAIN_SEED:-}"

# Optional convenience overrides
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
OUT_DIR="${OUT_DIR:-}"

if [[ -z "$EXP_DIR" && -z "$CFG_JSON" ]]; then
  echo "ERROR: Provide EXP_DIR or CFG_JSON via sbatch --export." >&2
  exit 2
fi

cmd=(python -m classifier_NN.legacy.run_experiments_A2 \
  --srs-root "$SRS_ROOT")

if [[ -n "$THRESHOLD" ]]; then
  cmd+=(--threshold "$THRESHOLD")
fi

if [[ -n "$TRAIN_LON_ABS_MAX" ]]; then
  cmd+=(--train-lon-abs-max "$TRAIN_LON_ABS_MAX")
fi
if [[ -n "$TRAIN_SEED" ]]; then
  cmd+=(--train-seed "$TRAIN_SEED")
fi

if [[ -n "$EXP_DIR" ]]; then
  cmd+=(--exp-dir "$EXP_DIR" --checkpoint "$CHECKPOINT")
fi
if [[ -n "$CFG_JSON" ]]; then
  cmd+=(--cfg-json "$CFG_JSON")
fi
if [[ -n "$CHECKPOINT_PATH" ]]; then
  cmd+=(--checkpoint-path "$CHECKPOINT_PATH")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  cmd+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$NUM_WORKERS" ]]; then
  cmd+=(--num-workers "$NUM_WORKERS")
fi
if [[ -n "$OUT_DIR" ]]; then
  cmd+=(--out-dir "$OUT_DIR")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
