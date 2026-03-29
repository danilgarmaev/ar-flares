#!/usr/bin/env bash
set -euo pipefail

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SCRIPT="${SCRIPT:-scripts/a1_2d_rerun_screen_224_a100.slurm}"

echo "Submitting A1 2D rerun screening matrix (10 jobs @ 224, seed 0)"
"$SBATCH_BIN" "$SCRIPT"
