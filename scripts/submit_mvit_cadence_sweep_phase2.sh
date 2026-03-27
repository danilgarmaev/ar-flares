#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "MViT Cadence Sweep: Phase 2"
echo "=========================================="
echo "Runs:"
echo "  cadences: 60, 120, 180"
echo "  labels: C+, M+"
echo "  seeds: 3, 4"
echo "  total: 12 runs"
echo ""

JOB_ID=$(sbatch --array=18-29 scripts/run_mvit_cadence_sweep_a100.slurm | awk '{print $4}')

echo "Submitted MViT cadence phase-2 array: $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"
