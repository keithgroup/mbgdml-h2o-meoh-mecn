#!/bin/bash

#SBATCH --job-name=3h2o-training
#SBATCH --output=3h2o-training.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --cluster=smp
#SBATCH --partition=smp

module purge

# Make sure you add ~/.local/bin to your .bashrc file.

export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR
/ihome/crc/install/python/miniconda3-3.7/bin/python mbgdml-model-train-crc.py

crc-job-stats.py
