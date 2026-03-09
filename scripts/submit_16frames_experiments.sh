#!/usr/bin/env bash
# Submit smoke test first, then full sweep
#
# Usage:
#   bash scripts/submit_16frames_experiments.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "16-Frame Cadence Experiments Submission"
echo "=========================================="
echo ""

# Check if we should run smoke test first
if [[ "${SKIP_SMOKE_TEST:-0}" == "0" ]]; then
    echo "Step 1: Submitting smoke test..."
    SMOKE_JOB=$(sbatch scripts/smoke_test_16frames_a100.slurm | awk '{print $4}')
    echo "  Smoke test job: $SMOKE_JOB"
    echo ""
    
    echo "Waiting for smoke test to complete (max 10 minutes)..."
    timeout 600 bash -c "while squeue -j $SMOKE_JOB -h >/dev/null 2>&1; do sleep 10; done" || {
        echo "  Warning: Timeout waiting for smoke test. Proceeding anyway..."
    }
    
    echo "  Checking smoke test results..."
    if ls results/*smoke* >/dev/null 2>&1; then
        echo "  ✅ Smoke test completed!"
    else
        echo "  ⚠️ Warning: No smoke test results found. Check logs/arfl-16f-smoke*.err"
        read -p "Continue with full sweep? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
    echo ""
else
    echo "Skipping smoke test (SKIP_SMOKE_TEST=1)"
    echo ""
fi

echo "Step 2: Submitting full sweep (6 runs: 2 models × 3 cadences)..."
SWEEP_JOB=$(sbatch scripts/run_16frames_cadence_sweep_a100.slurm | awk '{print $4}')
echo "  Full sweep job array: $SWEEP_JOB"
echo ""

echo "=========================================="
echo "Jobs submitted!"
echo "=========================================="
echo "Smoke test: $SMOKE_JOB (if run)"
echo "Full sweep:  $SWEEP_JOB"
echo ""
echo "Monitor with: watch -n 5 'squeue -u $USER'"
echo "Check logs:   tail -f logs/arfl-16f-cadence_${SWEEP_JOB}_*.err"
echo "Results:      ls -lht results/ | head -20"
echo "=========================================="
