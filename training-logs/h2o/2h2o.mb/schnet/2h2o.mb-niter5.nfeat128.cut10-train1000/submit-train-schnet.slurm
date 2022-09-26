#!/bin/bash
#SBATCH --job-name=2h2o-schnet-training
#SBATCH --output=2h2o-schnet-training.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=scavenger
#SBATCH --constraint=gtx1080

# Initialize conda environment
module purge
source activate /ihome/jkeith/amm503/miniconda3/envs/schnetpack-train

cd $SLURM_SUBMIT_DIR

python train-spk-schnet.py

crc-job-stats.py
