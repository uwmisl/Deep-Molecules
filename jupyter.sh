#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=ml4ml
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:6
#SBATCH --mem=300G
#SBATCH --time=240:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sidlak@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate /gscratch/ml4ml/sidlak/superfold/envs/pyroml

cat $0
echo "--------------------"

jupyter notebook --no-browser --ip 0.0.0.0 --port=8888
