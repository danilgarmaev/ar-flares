#!/bin/bash
#SBATCH --job-name=arfl-train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=def-jeandiro
# NOTE: If you have multiple accounts, change --account above to your preferred one

# ---- cluster-specific tweaks ----
# If your cluster uses constraints/partitions for A100/L40/etc, add one of:
#   #SBATCH --constraint=a100
#   #SBATCH --partition=gpu

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python/3.11
module load cuda/12.2

source /scratch/${USER}/venvs/ar-flares/bin/activate

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"

# Recommended: keep results on scratch (already default in config.py for Compute Canada)
export PYTHONUNBUFFERED=1

# Main training entrypoint
python -m classifier_NN.train

echo "Job finished at $(date)"






