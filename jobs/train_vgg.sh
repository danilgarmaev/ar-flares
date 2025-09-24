#!/bin/bash
#SBATCH --account=def-jeandiro 
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1             # request 1 GPU
#SBATCH --cpus-per-task=4        # matches DataLoader num_workers
#SBATCH --mem=32G                # adjust if needed
#SBATCH --time=06:00:00          # 4 hours, adjust as needed
#SBATCH --job-name=flare_vgg
#SBATCH --output=logs/%x-%j.out  # log file (%x = job name, %j = job ID)

module load python StdEnv/2020  gcc/11.3.0
module load cuda/11.8.0
module load cudnn

source /home/dgarmaev/envs/solar_eruptions/bin/activate
echo python_env_activated

# go to project directory
cd /home/dgarmaev/scratch/ar-flares

# run training
python classifier_VGG/model_training.py
