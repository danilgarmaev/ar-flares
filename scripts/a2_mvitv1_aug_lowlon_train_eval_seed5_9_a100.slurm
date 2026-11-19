#!/usr/bin/env bash
# Continue A2 longitude evaluation and central-only training for A1 MViT-v1
# seed 5..9 checkpoints. Submit after the matching A1 seed-offset array finishes:
#   sbatch --dependency=afterok:<A1_JOB_ID> scripts/a2_mvitv1_aug_lowlon_train_eval_seed5_9_a100.slurm

#SBATCH --job-name=arfl-a2mvitv1-low30-s5
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
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

idx="${SLURM_ARRAY_TASK_ID}"
if (( idx < 5 )); then
  LABEL_CLASS="C"
  SEED="$(( idx + 5 ))"
  NAME_SUFFIX="screen2dres_mvitv1_notemporal_cplus_dsnone300k_wbce_ftv2_aug"
else
  LABEL_CLASS="M"
  SEED="$(( idx ))"
  NAME_SUFFIX="screen2dres_mvitv1_notemporal_mplus_dsnone100k_wbce_ftv2_aug"
fi

mapfile -d '' MATCHES < <(
  find results -maxdepth 1 -type d \
    -name "*A1-224_mvit_seed${SEED}_${NAME_SUFFIX}" \
    -print0 | sort -z
)

if (( ${#MATCHES[@]} == 0 )); then
  echo "No completed A1-224 directory found for class=${LABEL_CLASS}+ seed=${SEED}" >&2
  exit 1
fi

EXP_DIR="${MATCHES[$(( ${#MATCHES[@]} - 1 ))]}"
if [[ ! -f "${EXP_DIR}/best_tss.pt" || ! -f "${EXP_DIR}/metrics.json" ]]; then
  echo "A1 directory is missing best_tss.pt or metrics.json: ${EXP_DIR}" >&2
  exit 1
fi

cmd=(python -m classifier_NN.legacy.run_experiments_A2
  --exp-dir "$EXP_DIR"
  --checkpoint best_tss.pt
  --srs-root /home/dgarmaev/scratch/ar-flares/data/SRS
  --train-lon-abs-max 30
  --train-seed "$SEED"
  --train-name-suffix "seed${SEED}_mvitv1_cont10"
)

echo "=========================================="
echo "A2 MViT-v1 continuation: baseline longitude eval + low-lon training"
echo "task=${idx} class=${LABEL_CLASS}+ seed=${SEED}"
echo "exp_dir=${EXP_DIR}"
echo "threshold will be inferred from metrics.json"
echo "Running: ${cmd[*]}"
echo "=========================================="
"${cmd[@]}"
