#!/usr/bin/env bash
#SBATCH --job-name=arfl-a1vgg
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=23:59:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Default to no-pretrained on compute nodes (no outbound internet). To enable pretrained:
#   sbatch --export=ALL,PRETRAINED=1 scripts/a1_vgg_full_a100.slurm
PRETRAINED=${PRETRAINED:-0}
EXTRA_PRETRAINED_ARGS="--no-pretrained"
if [ "$PRETRAINED" = "1" ]; then
	EXTRA_PRETRAINED_ARGS="--pretrained"
fi

python -m classifier_NN.legacy.run_experiments_A1_vgg_paper $EXTRA_PRETRAINED_ARGS
