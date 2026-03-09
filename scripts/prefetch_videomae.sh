#!/bin/bash
# Prefetch VideoMAE weights (run on login node with internet)
#
# This downloads the VideoMAE model weights to your cache directory
# so they're available for offline compute node runs.
#
# Usage:
#   bash scripts/prefetch_videomae.sh

set -e

echo "=========================================="
echo "Prefetching VideoMAE Weights"
echo "=========================================="
echo ""

# Activate environment
source /home/dgarmaev/envs/ar-flares/bin/activate

# Set cache location
export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Ensure we're online
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE
export OFFLINE=0

echo "Cache directory: $HF_HOME"
echo ""

echo "Checking transformers version..."
python -c "import transformers; print('transformers:', transformers.__version__)"
echo ""

echo "Attempting to download VideoMAE base model..."
python -c "
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

model_id = 'MCG-NJU/videomae-base'
print(f'Downloading {model_id}...')

try:
    model = VideoMAEForVideoClassification.from_pretrained(model_id)
    print(f'✅ Successfully downloaded {model_id}')
    print(f'   Model saved to cache: {model_id}')
except Exception as e:
    print(f'❌ Failed to download: {e}')
    print('   You may need to upgrade transformers:')
    print('   pip install --upgrade transformers>=4.30.0')
    exit(1)
"

echo ""
echo "=========================================="
echo "Prefetch complete!"
echo "=========================================="
echo ""
echo "Now you can run experiments in offline mode on compute nodes."
echo "The weights will be loaded from: $HF_HOME"
