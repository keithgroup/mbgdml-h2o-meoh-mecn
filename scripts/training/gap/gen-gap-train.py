import copy
import itertools
import os
import shutil

model_name = '2meoh.mb-gap'
xyz_train_path = 'meoh/2meoh/gap/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh-dset.mb-cm8-train.xyz'
xyz_valid_path = 'meoh/2meoh/gap/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh-dset.mb-cm8-valid.xyz'

write_dir = 'staging/'  # Relative to this Python file.
overwrite = True

n_sparse_grid = [4000]
n_max_grid = [8, 12, 16]
l_max_grid = [6, 12, 16]
cutoff_grid = [4.0, 8.0, 12.0, 16.0]
atom_sigma_grid = [0.5]  # Gaussian smearing width of atom density for SOAP, in Angstrom
delta_grid = [0.1]  # scaling of kernel, per descriptor, here for SOAP it is per atom, in eV
zeta_grid = [3]  # power kernel is raised to - together with dot_product gives a polynomial kernel
sigma_e_grid = [0.001]
sigma_f_grid = [0.01]




###   SCRIPT   ###
base_dir = '../../../'
ml_dset_dir = os.path.join(base_dir, 'data/ml-dsets/')
xyz_train_path = os.path.join(ml_dset_dir, xyz_train_path)
xyz_valid_path = os.path.join(ml_dset_dir, xyz_valid_path)

grid_params = [
    n_sparse_grid, n_max_grid, l_max_grid, cutoff_grid, atom_sigma_grid,
    delta_grid, zeta_grid, sigma_e_grid, sigma_f_grid
]

params_gen = itertools.product(*grid_params)

train_script = r"""#!/bin/bash
#SBATCH --job-name=JOB_NAME_VAL
#SBATCH --output=MODEL_NAME_VAL.out
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

train_path="train.xyz"
valid_path="valid.xyz"
model_name="MODEL_NAME_VAL"

cd $SLURM_SUBMIT_DIR
# Train GAP model
gap_fit at_file=$train_path \
gap={soap cutoff=CUTOFF_VAL \
    covariance_type=dot_product \
    zeta=ZETA_VAL \
    delta=DELTA_VAL \
    atom_sigma=ATOM_SIGMA_VAL \
    add_species=T \
    n_max=N_MAX_VAL \
    l_max=L_MAX_VAL \
    n_sparse=N_SPARSE_VAL \
    sparse_method=cur_points} \
force_parameter_name=forces \
e0_method=average \
default_sigma={SIGMA_E_VAL SIGMA_F_VAL 0.0 0.0} \
do_copy_at_file=F sparse_separate_file=T \
gp_file=$model_name.xml

# Validate GAP model
echo "Predicting GAP validation set"
time quip E=T F=T atoms_filename=$valid_path param_filename=$model_name.xml | grep AT | sed 's/AT//' > model-valid.xyz

echo "Computing validation statistics"
python -u get-gap-valid-error.py
echo "DONE"

crc-job-stats.py

"""

valid_script = r"""import ase
import ase.io
import csv
import numpy as np
import os

csv_name = 'model-valid-stats.csv'

os.chdir(os.path.dirname(os.path.realpath(__file__)))
valid_ref_path = 'valid.xyz'
valid_model_path = 'model-valid.xyz'
valid_ref = ase.io.read(valid_ref_path, ':')
valid_model = ase.io.read(valid_model_path, ':')

hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

# Energy data
E_true = np.array([at.get_potential_energy() for at in valid_ref])
E_pred = np.array([at.get_potential_energy() for at in valid_model])

# Force data
F_true, F_pred = [], []
for at_ref, at_pred in zip(valid_ref, valid_model):
    F_true.append(at_ref.get_forces())
    F_pred.append(at_pred.arrays['force'])
F_true = np.array(F_true)
F_pred = np.array(F_pred)

# Convert eV to kcal/mol
E_true *= ev2kcalmol
E_pred *= ev2kcalmol
F_true *= ev2kcalmol
F_pred *= ev2kcalmol

# Get errors
E_errors = E_pred - E_true
F_errors = F_pred - F_true

# Compute error statistics
E_mae = np.nanmean(np.abs(E_errors))
E_rmse = np.sqrt(np.nanmean((E_errors)**2))
E_max_error = np.nanmax(np.abs(E_errors))
E_sse = np.dot(E_errors[~np.isnan(E_errors)], E_errors[~np.isnan(E_errors)])
F_mae = np.nanmean(np.abs(F_errors))
F_rmse = np.sqrt(np.nanmean((F_errors)**2))
F_max_error = np.nanmax(np.abs(F_errors))
F_sse = np.dot(F_errors[~np.isnan(F_errors)], F_errors[~np.isnan(F_errors)])

csv_data = [
    ['E_unit', 'R_unit', 'E_mae', 'E_rmse', 'E_sse', 'E_max_abs_err', 'F_mae', 'F_rmse', 'F_sse', 'F_max_abs_err'],
    ['kcal/mol', 'Angstrom', E_mae, E_rmse, E_sse, E_max_error, F_mae, F_rmse, F_sse, F_max_error]
]

with open(csv_name, 'w', encoding='utf-8') as f_csv:
    csv_writer = csv.writer(f_csv)
    csv_writer.writerows(csv_data)

"""

os.chdir(os.path.dirname(os.path.realpath(__file__)))
i = 0
for params in params_gen:
    dir_name = model_name + f'-{i}'
    dir_path = os.path.join(write_dir, dir_name)
    n_sparse, n_max, l_max, cutoff, atom_sigma, delta, zeta, sigma_e, sigma_f = params
    
    os.makedirs(dir_path, exist_ok=overwrite)

    train_script_job = copy.copy(train_script)
    train_script_job = train_script_job.replace('JOB_NAME_VAL', dir_name)
    train_script_job = train_script_job.replace('MODEL_NAME_VAL', model_name)
    train_script_job = train_script_job.replace('N_SPARSE_VAL', str(n_sparse))
    train_script_job = train_script_job.replace('N_MAX_VAL', str(n_max))
    train_script_job = train_script_job.replace('L_MAX_VAL', str(l_max))
    train_script_job = train_script_job.replace('CUTOFF_VAL', str(cutoff))
    train_script_job = train_script_job.replace('ATOM_SIGMA_VAL', str(atom_sigma))
    train_script_job = train_script_job.replace('DELTA_VAL', str(delta))
    train_script_job = train_script_job.replace('ZETA_VAL', str(zeta))
    train_script_job = train_script_job.replace('SIGMA_E_VAL', str(sigma_e))
    train_script_job = train_script_job.replace('SIGMA_F_VAL', str(sigma_f))

    # Copy and write files
    with open(os.path.join(dir_path, 'submit-gap-train.slurm'), 'w', encoding='utf-8') as f:
        f.write(train_script_job)
    with open(os.path.join(dir_path, 'get-gap-valid-error.py'), 'w', encoding='utf-8') as f:
        f.write(valid_script)
    shutil.copyfile(xyz_train_path, os.path.join(dir_path, 'train.xyz'))
    shutil.copyfile(xyz_valid_path, os.path.join(dir_path, 'valid.xyz'))

    i += 1
