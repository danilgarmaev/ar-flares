#!/usr/bin/env bash
#SBATCH --job-name=arfl-video-agg
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
source /home/dgarmaev/envs/ar-flares/bin/activate
python scripts/aggregate_video_summary.py
