#!/usr/bin/env bash
#SBATCH --job-name=arfl-a3-report
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

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

MODE="${TARGET_MODE:-all}"
FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_FLAG+=(--force)
fi

echo "=========================================="
echo "A3 report reevaluation"
echo "  mode=${MODE} task=${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

python scripts/reevaluate_a3_report.py   --results-dir results   --mode "$MODE"   --task-id "$SLURM_ARRAY_TASK_ID"   "${FORCE_FLAG[@]}"
