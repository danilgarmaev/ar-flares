#!/usr/bin/env bash
# Evaluate the currently best low-longitude A2 checkpoints on longitude buckets
# without interrupting the ongoing training jobs.

#SBATCH --job-name=arfl-a2mvit-loev
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-9

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

module load gcc opencv
source /home/dgarmaev/envs/ar-flares/bin/activate

export AR_FLARES_WDS_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_flow
export AR_FLARES_RESULTS_BASE=/home/dgarmaev/scratch/ar-flares/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/home/dgarmaev/scratch/ar-flares/data

export OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

EXP_DIRS=(
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 12:49:43_A2-lowLon30_2026-03-29 18:50:42_A1-224_mvitv2_small_seed0_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug_seed0"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 12:53:19_A2-lowLon30_2026-03-29 18:53:41_A1-224_mvitv2_small_seed1_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug_seed1"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 12:53:19_A2-lowLon30_2026-03-29 18:54:44_A1-224_mvitv2_small_seed2_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug_seed2"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 12:49:43_A2-lowLon30_2026-03-29 18:53:41_A1-224_mvitv2_small_seed3_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug_seed3"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 12:49:43_A2-lowLon30_2026-03-29 18:56:41_A1-224_mvitv2_small_seed4_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug_seed4"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 12:49:43_A2-lowLon30_2026-03-29 19:05:41_A1-224_mvitv2_small_seed0_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug_seed0"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 13:14:11_A2-lowLon30_2026-03-29 19:05:36_A1-224_mvitv2_small_seed1_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug_seed1"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 13:28:12_A2-lowLon30_2026-03-29 19:05:36_A1-224_mvitv2_small_seed2_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug_seed2"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 13:32:16_A2-lowLon30_2026-03-29 19:05:47_A1-224_mvitv2_small_seed3_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug_seed3"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-30 13:32:17_A2-lowLon30_2026-03-29 19:05:47_A1-224_mvitv2_small_seed4_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug_seed4"
)

idx="${SLURM_ARRAY_TASK_ID}"
EXP_DIR="${EXP_DIRS[$idx]}"
EPOCH_METRICS_PATH="${EXP_DIR}/epoch_metrics.jsonl"

THRESHOLD="$(/home/dgarmaev/envs/ar-flares/bin/python - <<PY
import json
best=None
with open(r'''${EPOCH_METRICS_PATH}''') as f:
    for line in f:
        r=json.loads(line)
        t=r.get('val_best_tss')
        th=r.get('val_best_threshold')
        if t is None or th is None:
            continue
        if best is None or float(t) > best[0]:
            best=(float(t), float(th))
if best is None:
    raise SystemExit('No val_best_tss / val_best_threshold found')
print(best[1])
PY
)"

cmd=(python -m classifier_NN.legacy.run_experiments_A2
  --exp-dir "$EXP_DIR"
  --checkpoint best_tss.pt
  --srs-root /home/dgarmaev/scratch/ar-flares/data/SRS
  --threshold "$THRESHOLD"
)

echo "=========================================="
echo "A2 low-longitude current-best longitude evaluation"
echo "=========================================="
echo "task=$idx"
echo "exp_dir=$EXP_DIR"
echo "threshold=$THRESHOLD"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
