#!/usr/bin/env bash
# Evaluate the 5-seed A1 224+aug MViT baseline runs by longitude bucket for C+ and M+.

#SBATCH --job-name=arfl-a2mvit-eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:50:42_A1-224_mvitv2_small_seed0_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:53:41_A1-224_mvitv2_small_seed1_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:54:44_A1-224_mvitv2_small_seed2_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:53:41_A1-224_mvitv2_small_seed3_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 18:56:41_A1-224_mvitv2_small_seed4_screen2dres_mvit_cplus_dsnone300k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:05:41_A1-224_mvitv2_small_seed0_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:05:36_A1-224_mvitv2_small_seed1_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:05:36_A1-224_mvitv2_small_seed2_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:05:47_A1-224_mvitv2_small_seed3_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
  "/home/dgarmaev/scratch/ar-flares/results/2026-03-29 19:05:47_A1-224_mvitv2_small_seed4_screen2dres_mvit_mplus_dsnone100k_wbce_ftv2_aug"
)
THRESHOLDS=(
  0.3720894943661988
  0.1646989409425296
  0.392
  0.181
  0.2044328810321167
  0.2002280047358945
  0.06176485449267784
  0.006986619509252208
  0.2616887108343071
  0.047824624604196286
)

idx="${SLURM_ARRAY_TASK_ID}"
EXP_DIR="${EXP_DIRS[$idx]}"
THRESHOLD="${THRESHOLDS[$idx]}"

cmd=(python -m classifier_NN.legacy.run_experiments_A2
  --exp-dir "$EXP_DIR"
  --checkpoint best_tss.pt
  --srs-root /home/dgarmaev/scratch/ar-flares/data/SRS
  --threshold "$THRESHOLD"
)

echo "=========================================="
echo "A2 baseline longitude evaluation"
echo "=========================================="
echo "task=$idx"
echo "exp_dir=$EXP_DIR"
echo "threshold=$THRESHOLD"
echo "Running: ${cmd[*]}"
"${cmd[@]}"
