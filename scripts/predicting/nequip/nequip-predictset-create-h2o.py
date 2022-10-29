# MIT License
# 
# Copyright (c) 2022, Alex M. Maldonado
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Creates a prediction set"""

import csv
import numpy as np
import os
from reptar import File
import time
import torch

from nequip.data import AtomicData, AtomicDataDict
from nequip.scripts.train import check_code_version
from nequip.utils._global_options import _set_global_options
from nequip.train import Trainer
from nequip.utils import Config
from nequip.data.transforms import TypeMapper

overwrite = True
save = True
theory = 'mp2.def2tzvp.frozencore'
r_unit = 'Angstrom'
e_unit = 'kcal/mol'

r_cutoff = 8  # must be int (for file paths)
csv_name = f'h2o-psets-nequip.cut{r_cutoff}.train1000.csv'
csv_dir = 'data/psets/h2o/'
csv_headers = [
    'dset_key', 'group_key', 'E_unit', 'R_unit',
    'E_mae', 'E_rmse', 'E_sse', 'E_max_abs_err', 'F_mae', 'F_rmse', 'F_sse', 'F_max_abs_err',
    'time'
]

in_ev = False  # If model is in eV. If False, we assume it is in kcal/mol.
use_ray = False
n_cores = None

jobs = [
    # Training sets
    #{'dset_path': 'md-sampling/140h2o-xtb.md-samples.exdir', 'group_key': '1h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'md-sampling/140h2o-xtb.md-samples.exdir', 'group_key': '2h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'md-sampling/140h2o-xtb.md-samples.exdir', 'group_key': '3h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    # 4mer isomers
    {'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '4h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '4h2o/samples_1h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '4h2o/samples_2h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '4h2o/samples_3h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    # 5mer isomers
    {'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '5h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '5h2o/samples_1h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '5h2o/samples_2h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '5h2o/samples_3h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    # 6mer isomers
    {'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '6h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '6h2o/samples_1h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '6h2o/samples_2h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/h2o-temelso.etal.exdir', 'group_key': '6h2o/samples_3h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    # 16mer
    {'dset_path': 'isomers/16h2o-yoo.etal.exdir', 'group_key': '', 'E_key': 'energy_ele_rimp2.def2tzvp_orca', 'G_key': 'grads_rimp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/16h2o-yoo.etal.exdir', 'group_key': 'samples_1h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/16h2o-yoo.etal.exdir', 'group_key': 'samples_2h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
    #{'dset_path': 'isomers/16h2o-yoo.etal.exdir', 'group_key': 'samples_3h2o', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'G_key': 'grads_mp2.def2tzvp_orca'},
]

config_path = f'h2o/nequip/train_1000-lr_0.01-cut_{r_cutoff}/config.yaml'
model_path = f'h2o/nequip/train_1000-lr_0.01-cut_{r_cutoff}/best_model.pth'
tm = TypeMapper(chemical_symbol_to_type={"H": 0, "C": 1, "N": 2, "O": 3})
device = 'cpu'


###   SCRIPT   ###


hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

os.chdir(os.path.dirname(os.path.realpath(__file__)))
base_dir = '../../../'
dset_dir = os.path.join(base_dir, 'data/')
model_dir = '../../../../mbgdml-h2o-meoh-mecn-models'

config_path = os.path.join(model_dir, config_path)
model_path = os.path.join(model_dir, model_path)

# Load model
global_config = Config.from_file(config_path)
_set_global_options(global_config)
check_code_version(global_config)
del global_config
model_path_dir = os.path.dirname(model_path)
model, model_config = Trainer.load_model_from_training_session(
    traindir=model_path_dir, model_name='best_model.pth'
)

model = model.to(device)
model_r_max = model_config["r_max"]
model = model.eval()

csv_data = [csv_headers]
for job in jobs:
    dset_path = job['dset_path']
    group_key = job['group_key']

    data_keys = {
        'Z': f'{group_key}/atomic_numbers',
        'R': f'{group_key}/geometry',
        'E': f'{group_key}/{job["E_key"]}',
        'G': f'{group_key}/{job["G_key"]}',
    }

    csv_data.append([dset_path, group_key])


    dset_path = os.path.join(dset_dir, dset_path)

    # Preparing the data set.
    rfile = File(dset_path)
    Z = rfile.get(data_keys['Z'])
    R = rfile.get(data_keys['R'])
    E_true = rfile.get(data_keys['E'])
    F_true = -rfile.get(data_keys['G'])

    if R.ndim == 2:
        R = R[..., None]
    if E_true.ndim == 0:
        E_true = np.array([E_true])
    if F_true.ndim == 2:
        F_true = F_true[..., None]
    
    if in_ev:
        E_true *= hartree2ev
        F_true *= hartree2ev
    else:
        E_true *= hartree2kcalmol
        F_true *= hartree2kcalmol

    t_start = time.time()
    # Creating the predict set.
    atomic_numbers = torch.Tensor(
            torch.from_numpy(Z.astype(np.float32))
    ).to(torch.int64)

    atom_types = tm.transform(atomic_numbers)

    atomic_data = {
        AtomicDataDict.ATOMIC_NUMBERS_KEY: atomic_numbers,
        AtomicDataDict.ATOM_TYPE_KEY: atom_types
    }

    print('Making predictions')
    E_pred = np.empty(E_true.shape)
    F_pred = np.empty(F_true.shape)

    R = torch.from_numpy(R)
    for i in range(len(R)):
        data = AtomicData.from_points(
            pos=R[i], r_max=model_r_max, **atomic_data
        )
        pred = model(AtomicData.to_AtomicDataDict(data))

        E_pred[i] = pred['total_energy'].detach().numpy()[0]
        F_pred[i] = pred['forces'].detach().numpy()
    t_stop = time.time()
    t_pred = t_stop - t_start

    print('\nResults')
    # Compute energy and force prediction errors
    E_errors = E_pred - E_true
    F_errors = F_pred.flatten() - F_true.flatten()
    if in_ev:
        E_errors *= ev2kcalmol
        F_errors *= ev2kcalmol
    
    # Compute statistics
    E_mae = np.nanmean(np.abs(E_errors))
    E_rmse = np.sqrt(np.nanmean((E_errors)**2))
    E_max_error = np.nanmax(np.abs(E_errors))
    E_sse = np.dot(E_errors[~np.isnan(E_errors)], E_errors[~np.isnan(E_errors)])
    F_mae = np.nanmean(np.abs(F_errors))
    F_rmse = np.sqrt(np.nanmean((F_errors)**2))
    F_max_error = np.nanmax(np.abs(F_errors))
    F_sse = np.dot(F_errors[~np.isnan(F_errors)], F_errors[~np.isnan(F_errors)])
    print(f'Energy MAE:  {E_mae:.5f} kcal/mol')
    print(f'Energy RMSE: {E_rmse:.5f} kcal/mol')
    print(f'Energy SSE: {E_sse:.5f} [kcal/mol]^2')
    print(f'Energy max: {E_max_error:.5f} kcal/mol')
    print(f'Force MAE:  {F_mae:.5f} kcal/(mol A)')
    print(f'Force RMSE: {F_rmse:.5f} kcal/(mol A)')
    print(f'Force SSE: {F_sse:.5f} [kcal/(mol A)]^2')
    print(f'Force max: {F_max_error:.5f} kcal/(mol A)')

    # Writing csv
    if save:
        csv_data[-1].extend(
            [
                'kcal/mol', 'Angstrom', E_mae, E_rmse, E_sse, E_max_error,
                F_mae, F_rmse, F_sse, F_max_error, t_pred
            ]
        )

        with open(os.path.join(base_dir, csv_dir, csv_name), 'w') as f_csv:
            csv_writer = csv.writer(f_csv)
            csv_writer.writerows(csv_data)
