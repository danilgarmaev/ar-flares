#!/bin/bash
#
# Full sweep: Run R(2+1)D with all 3 cadences (36, 72, 108 min) and 3 seeds
#
# This matches the macros project methodology:
# - 16 frames per sequence
# - 36 min cadence = 9 hours total (3 steps * 12 min/step)
# - 72 min cadence = 18 hours total (6 steps * 12 min/step) <- matches macros
# - 108 min cadence = 27 hours total (9 steps * 12 min/step)
#
# Usage:
#   bash scripts/run_16frames_cadence_sweep.sh
#

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "16-Frame Cadence Sweep"
echo "=========================================="
echo "Model: R(2+1)D"
echo "Cadences: 36, 72, 108 minutes"
echo "Seeds: 0, 1, 2"
echo "Total runs: 9"
echo "=========================================="
echo ""

python -m classifier_NN.legacy.run_experiments_16frames_cadence \
    --backbone r2plus1d_18 \
    --interval-min 36 \
    --interval-min 72 \
    --interval-min 108 \
    --n-seeds 3 \
    --epochs 50 \
    --pretrained-3d

echo ""
echo "=========================================="
echo "Full sweep complete!"
echo "Check results/summary_16frames_cadence_sweep.json for aggregated results"
echo "=========================================="
