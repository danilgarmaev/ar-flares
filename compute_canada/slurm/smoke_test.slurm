#!/bin/bash
#SBATCH --job-name=arfl-smoke
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --account=def-jeandiro
# NOTE: If you have multiple accounts, change --account above to your preferred one

set -euo pipefail

echo "Smoke test started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python/3.11
module load cuda/12.2

source /scratch/${USER}/venvs/ar-flares/bin/activate

export PYTHONUNBUFFERED=1

python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

echo "Running smoke test..."
timeout 180 python -m classifier_NN.smoke_test

echo "Smoke test finished at $(date)"
