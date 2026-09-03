#!/bin/bash
#SBATCH --job-name=best_hparam_adam
#SBATCH --output=logs_best_hparams/best_hparam_adam.log
#SBATCH --gpus=1
#SBATCH --time=48:00:00

module load cuda/12.6
source ~/miniforge3/bin/activate gpt2

cd "$(dirname "$0")/.."

python3 -u best_hparams/sweep_adam.py
