#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "A3 Transformer Diagnostic Sweep"
echo "=========================================="
echo "Smoke: MViT C+ 108min sanity run"
echo "Full: 8 runs, 2 GPUs each, 20h wall time"
echo ""
echo "Reminder: prefetch MViT weights first on a login node if needed:"
echo "  source /home/dgarmaev/envs/ar-flares/bin/activate"
echo "  python scripts/prefetch_a3_video_backbones.py --mvit --mvit-model-id mvit_v1_b"
echo ""

SMOKE_JOB=$(sbatch scripts/smoke_a3_transformer_diagnostics_a100.slurm | awk '{print $4}')
FULL_JOB=$(sbatch --dependency=afterok:${SMOKE_JOB} scripts/run_a3_transformer_diagnostics_a100.slurm | awk '{print $4}')

echo "Submitted smoke job: $SMOKE_JOB"
echo "Submitted dependent full array: $FULL_JOB"
echo "Monitor with: squeue -j $SMOKE_JOB,$FULL_JOB"
