#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "VideoSwin DP vs DDP Smoke Benchmark"
echo "=========================================="
echo "Config: weighted_bce, M+, 72min, noaug/downsampled"
echo "DP:  batch=8 global on 2 GPUs"
echo "DDP: batch=4 per GPU on 2 GPUs (global 8)"
echo "Smoke cap: steps_per_epoch=512, val_max_batches=64"
echo ""

DP_JOB=$(sbatch scripts/smoke_videoswin_16frames_weighted_bce_dp_benchmark_a100.slurm | awk '{print $4}')
DDP_JOB=$(sbatch scripts/smoke_videoswin_16frames_weighted_bce_ddp_benchmark_a100.slurm | awk '{print $4}')

echo "Submitted DP benchmark job:  $DP_JOB"
echo "Submitted DDP benchmark job: $DDP_JOB"
echo "Monitor with: squeue -j $DP_JOB,$DDP_JOB"
echo "Logs:"
echo "  logs/arfl-vswin-dp-bmk_${DP_JOB}.out / .err"
echo "  logs/arfl-vswin-ddp-bm_${DDP_JOB}.out / .err"
