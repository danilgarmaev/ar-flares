#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CADENCE="${CADENCE:-120}"
SEED="${SEED:-0}"
MVIT_MODEL_ID="${MVIT_MODEL_ID:-mvit_v1_b}"

echo "=========================================="
echo "MViT Full-Data Ablation"
echo "=========================================="
echo "Run:"
echo "  label: C+"
echo "  cadence: ${CADENCE}"
echo "  seed: ${SEED}"
echo ""

JOB_ID=$(
  sbatch \
    --job-name=arfl-mvit-full \
    --gres=gpu:a100:2 \
    --cpus-per-task=12 \
    --mem=64G \
    --time=20:00:00 \
    --account=def-jeandiro \
    --output=logs/%x_%j.out \
    --error=logs/%x_%j.err \
    --wrap "cd /home/dgarmaev/scratch/ar-flares && \
source /home/dgarmaev/envs/ar-flares/bin/activate && \
export PYTHONUNBUFFERED=1 && \
export AR_FLARES_WDS_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_out && \
export AR_FLARES_WDS_FLOW_BASE=/home/dgarmaev/scratch/ar-flares/data/wds_flow && \
export AR_FLARES_RESULTS_BASE=/home/dgarmaev/scratch/ar-flares/results && \
export AR_FLARES_INTENSITY_LABELS_ROOT=/home/dgarmaev/scratch/ar-flares/data && \
export OFFLINE=1 && \
export HF_DATASETS_OFFLINE=1 && \
export TRANSFORMERS_OFFLINE=1 && \
export HF_HUB_OFFLINE=1 && \
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf && \
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch && \
mkdir -p logs results \"\$HF_HOME\" \"\$TORCH_HOME\" && \
python -m classifier_NN.legacy.run_experiments_16frames_cadence \
  --backbone mvit \
  --interval-min ${CADENCE} \
  --seed ${SEED} \
  --single \
  --epochs 50 \
  --batch-size 8 \
  --image-size 224 \
  --lr 1e-4 \
  --backbone-lr 1e-6 \
  --head-lr 1e-4 \
  --warmup-epochs 3 \
  --grad-clip-norm 1.0 \
  --min-flare-class C \
  --loss-type weighted_bce \
  --scheduler cosine \
  --early-stopping-min-epoch 3 \
  --threshold-min-precision 0.15 \
  --threshold-min-recall 0.5 \
  --num-workers 8 \
  --use-multi-gpu \
  --multi-gpu-max-devices 2 \
  --pretrained-3d \
  --pretrained-mvit \
  --mvit-model-id ${MVIT_MODEL_ID} \
  --no-balance-classes \
  --no-balanced-batch-sampling \
  --no-aug \
  --run-tag full_cad${CADENCE}" | awk '{print $4}'
)

echo "Submitted MViT full-data ablation: $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"
