#!/usr/bin/env bash
# Re-evaluate the current best resumed A1 MViT rescue checkpoints on the test set.

#SBATCH --job-name=arfl-a1mvit-eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=def-jeandiro
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /home/dgarmaev/scratch/ar-flares
mkdir -p logs results

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
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:59:50_A1-56_mvitv2_small_seed1_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:13:40_A1-112_mvitv2_small_seed1_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:14:21_A1-112_mvitv2_small_seed3_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:14:41_A1-112_mvitv2_small_seed4_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:18:23_A1-56_mvitv2_small_seed4_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
)
CHECKPOINTS=(
  "resume_to15_mvit_cplus_56_seed1_aug.pt"
  "resume_to17_mvit_mplus_112_seed1_aug.pt"
  "resume_to14_mvit_mplus_112_seed3_aug.pt"
  "resume_to16_mvit_mplus_112_seed4_aug.pt"
  "resume_to17_mvit_mplus_56_seed4_aug.pt"
)
EPOCH_METRICS=(
  "epoch_metrics_resume_to15_mvit_cplus_56_seed1_aug.jsonl"
  "epoch_metrics_resume_to17_mvit_mplus_112_seed1_aug.jsonl"
  "epoch_metrics_resume_to14_mvit_mplus_112_seed3_aug.jsonl"
  "epoch_metrics_resume_to16_mvit_mplus_112_seed4_aug.jsonl"
  "epoch_metrics_resume_to17_mvit_mplus_56_seed4_aug.jsonl"
)
SUFFIXES=(
  "fixedthr_testeval_resume_to15_mvit_cplus_56_seed1_aug"
  "fixedthr_testeval_resume_to17_mvit_mplus_112_seed1_aug"
  "fixedthr_testeval_resume_to14_mvit_mplus_112_seed3_aug"
  "fixedthr_testeval_resume_to16_mvit_mplus_112_seed4_aug"
  "fixedthr_testeval_resume_to17_mvit_mplus_56_seed4_aug"
)

idx="${SLURM_ARRAY_TASK_ID}"
EXP_DIR="${EXP_DIRS[$idx]}"
CHECKPOINT="${CHECKPOINTS[$idx]}"
EPOCH_METRICS_PATH="${EXP_DIR}/${EPOCH_METRICS[$idx]}"
SUFFIX="${SUFFIXES[$idx]}"

FIXED_THRESHOLD="$(/home/dgarmaev/envs/ar-flares/bin/python - <<PY
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

cmd=(python -m classifier_NN.reevaluate_checkpoint
  "$EXP_DIR"
  --checkpoint "$CHECKPOINT"
  --suffix "$SUFFIX"
  --fixed-threshold "$FIXED_THRESHOLD"
)

echo "=========================================="
echo "A1 rescue eval from current resumed best checkpoint"
echo "=========================================="
echo "task=$idx"
echo "exp_dir=$EXP_DIR"
echo "checkpoint=$CHECKPOINT"
echo "fixed_threshold=$FIXED_THRESHOLD"
echo "suffix=$SUFFIX"
echo "Running: ${cmd[*]}"
"${cmd[@]}"

