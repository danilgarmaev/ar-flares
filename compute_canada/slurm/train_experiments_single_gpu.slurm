#!/bin/bash
#SBATCH --job-name=arfl-exps
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=def-jeandiro
# NOTE: If you have multiple accounts, change --account above to your preferred one

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd /home/${USER}/ar-flares

module load python/3.11
module load cuda/12.2
source /scratch/${USER}/venvs/ar-flares/bin/activate

export PYTHONUNBUFFERED=1

# Runs a sequence of experiments defined in classifier_NN/run_experiments.py
python -m classifier_NN.run_experiments

echo "Job finished at $(date)"


