#!/usr/bin/env bash
#SBATCH --job-name=arfl-a1vgg-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
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

# Put caches on scratch so they can be reused across jobs (compute nodes often have no internet).
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

python -m classifier_NN.legacy.run_experiments_A1_vgg_paper --smoke --no-pretrained
