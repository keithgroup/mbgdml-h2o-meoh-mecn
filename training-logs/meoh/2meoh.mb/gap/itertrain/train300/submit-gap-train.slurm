#!/bin/bash
#SBATCH --job-name=2meoh.mb-gap-itrain200
#SBATCH --output=2meoh.mb-gap-itrain200.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=1-00:00:00
#SBATCH --cluster=smp
#SBATCH --partition=smp

# Initialize conda environment
module purge
source activate /ihome/jkeith/amm503/miniconda3/envs/gap-train

export PATH=$PATH:/ihome/jkeith/amm503/codes/QUIP/build/linux_x86_64_gfortran_openmp
export OMP_NUM_THREADS=1
ulimit -s unlimited
export OMP_STACKSIZE=5G

train_path="train-300.xyz"
model_name="2meoh.mb-gap"

cd $SLURM_SUBMIT_DIR
# Train GAP model
gap_fit at_file=$train_path \
gap={soap cutoff=4.0 \
    covariance_type=dot_product \
    zeta=3 \
    delta=0.1 \
    atom_sigma=0.5 \
    add_species=T \
    n_max=8 \
    l_max=6 \
    n_sparse=4000 \
    sparse_method=cur_points} \
force_parameter_name=forces \
e0_method=average \
default_sigma={0.001 0.01 0.0 0.0} \
do_copy_at_file=F sparse_separate_file=T \
gp_file=$model_name.xml

echo "Getting next training set"
python -u get-next-gap-train-set.py
echo "DONE"

crc-job-stats.py

