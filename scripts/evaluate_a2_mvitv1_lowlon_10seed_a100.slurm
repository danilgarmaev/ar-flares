#!/usr/bin/env bash
# Evaluation-only pass for the 10-seed A2 central-only MViT-v1 checkpoints.
# This recomputes longitude-bucket metrics, including AUC/PR-AUC arrays that
# were not retained by the original train+eval run.

#SBATCH --job-name=arfl-a2mvitv1-loev10
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-19

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

module load gcc opencv
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

mapfile -d '' EXP_DIRS < <(
  find results -maxdepth 1 -type d \
    -name '2026-06-11 *A2-lowLon30_*mvitv1_10seed' \
    -print0 | sort -z
)

if [[ "${#EXP_DIRS[@]}" -ne 20 ]]; then
  echo "ERROR: expected 20 A2 low-lon MViT-v1 10-seed directories, found ${#EXP_DIRS[@]}" >&2
  printf '%s\n' "${EXP_DIRS[@]}" >&2
  exit 2
fi

idx="${SLURM_ARRAY_TASK_ID}"
EXP_DIR="${EXP_DIRS[$idx]}"

cmd=(python -m classifier_NN.legacy.run_experiments_A2
  --exp-dir "$EXP_DIR"
  --checkpoint best_tss.pt
  --srs-root /home/dgarmaev/scratch/ar-flares/data/SRS
)

echo "=========================================="
echo "A2 MViT-v1 central-only evaluation-only pass"
echo "=========================================="
echo "task=$idx"
echo "exp_dir=$EXP_DIR"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
