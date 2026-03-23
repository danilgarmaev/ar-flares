#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "VideoSwin 16-Frame Clean Sweep Submission"
echo "=========================================="
echo "Smoke: 1 run, 2 GPUs, DataParallel"
echo "Full: 12 runs, 2 GPUs each, dedicated CPUs, 36h wall time"
echo ""

SMOKE_JOB=$(sbatch scripts/smoke_videoswin_16frames_clean_a100.slurm | awk '{print $4}')
FULL_JOB=$(sbatch --dependency=afterok:${SMOKE_JOB} scripts/run_videoswin_16frames_clean_a100.slurm | awk '{print $4}')

echo "Submitted smoke job: $SMOKE_JOB"
echo "Submitted dependent full array: $FULL_JOB"
echo "Monitor with: squeue -j $SMOKE_JOB,$FULL_JOB"
echo "Smoke logs: logs/arfl-vswin-smoke_${SMOKE_JOB}.out / .err"
echo "Full logs:  logs/arfl-vswin-clean_${FULL_JOB}_*.out / .err"
echo ""
echo "Overrides if needed:"
echo "  sbatch --export=ALL,NUM_WORKERS=10,BATCH_SIZE=8,EPOCHS=50 scripts/run_videoswin_16frames_clean_a100.slurm"
