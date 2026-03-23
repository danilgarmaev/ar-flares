#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/dgarmaev/envs/ar-flares/bin/activate

MODE="${TARGET_MODE:-all}"
FORCE="${FORCE:-0}"
COUNT=$(python scripts/reevaluate_a3_report.py --results-dir results --mode "$MODE" --count)

if [[ "$COUNT" == "0" ]]; then
  echo "No A3 runs matched mode=$MODE"
  exit 0
fi

ARRAY_SPEC="0-$((COUNT - 1))"
REEVAL_JOB=$(sbatch --array="$ARRAY_SPEC" --export=ALL,TARGET_MODE="$MODE",FORCE="$FORCE" scripts/reevaluate_a3_report_a100.slurm | awk '{print $4}')
AGG_JOB=$(sbatch --dependency=afterok:${REEVAL_JOB} scripts/aggregate_video_summary_cpu.slurm | awk '{print $4}')

echo "Submitted A3 report reevaluation array: $REEVAL_JOB"
echo "  mode=$MODE tasks=$COUNT array=$ARRAY_SPEC"
echo "Submitted dependent summary aggregation job: $AGG_JOB"
echo "Monitor with: squeue -j $REEVAL_JOB,$AGG_JOB"
