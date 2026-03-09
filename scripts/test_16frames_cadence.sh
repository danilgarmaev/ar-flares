#!/bin/bash
#
# Quick test: Run R(2+1)D with one cadence (72 min = 18 hours, matching macros)
#
# Usage:
#   bash scripts/test_16frames_cadence.sh
#

cd "$(dirname "$0")/.." || exit 1

echo "Testing R(2+1)D with 16 frames, 72-minute cadence (18 hours total)"
echo "This is a quick test with limited epochs to verify the setup..."
echo ""

python -m classifier_NN.legacy.run_experiments_16frames_cadence \
    --backbone r2plus1d_18 \
    --interval-min 72 \
    --seed 0 \
    --single \
    --epochs 5 \
    --steps-per-epoch 100 \
    --val-max-batches 50 \
    --pretrained-3d

echo ""
echo "Quick test complete! Check results/ directory for output."
