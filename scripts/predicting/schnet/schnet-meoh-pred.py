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

"""Make normal schnet predictions"""

import ase
import csv
import numpy as np
import os
from reptar import File
import time
import torch
from mbgdml.models import schnetModel
from mbgdml.predictors import predict_schnet
import schnetpack

overwrite = True
save = False
theory = 'mp2.def2tzvp.frozencore'
r_unit = 'Angstrom'
e_unit = 'eV'

csv_name = 'meoh-psets-schnet.standard.csv'
csv_dir = 'data/psets/meoh/'
csv_headers = [
    'dset_key', 'group_key', '1-body', '2-body', '3-body', 'E_unit', 'R_unit',
    'E_mae', 'E_rmse', 'E_sse', 'E_max_abs_err', 'F_mae', 'F_rmse', 'F_sse', 'F_max_abs_err',
    'time'
]

in_ev = True  # If model is in eV. If False, we assume it is in kcal/mol.
use_ray = False
n_cores = None

jobs = [
    # Training sets
    # 4mer isomers
    {'dset_path': 'isomers/meoh-boyd.etal.exdir', 'group_key': '4meoh', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'F_key': 'grads_mp2.def2tzvp_orca', 'save_dir': 'data/psets/meoh/gap', 'name': '4meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000'},
    # 5mer isomers
    {'dset_path': 'isomers/meoh-boyd.etal.exdir', 'group_key': '5meoh', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'F_key': 'grads_mp2.def2tzvp_orca', 'save_dir': 'data/psets/meoh/gap', 'name': '5meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000'},
    # 6mer isomers
    {'dset_path': 'isomers/meoh-boyd.etal.exdir', 'group_key': '6meoh', 'E_key': 'energy_ele_mp2.def2tzvp_orca', 'F_key': 'grads_mp2.def2tzvp_orca', 'save_dir': 'data/psets/meoh/gap', 'name': '6meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000'},
    # 16mer
    {'dset_path': 'isomers/16meoh-pires.deturi.exdir', 'group_key': '', 'E_key': 'energy_ele_rimp2.def2tzvp_orca', 'F_key': 'grads_rimp2.def2tzvp_orca', 'save_dir': 'data/psets/meoh/gap', 'name': '16meoh.pires.deturi-pset-62meoh.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000'},
]

model_path = 'meoh/meoh-model-schnet.niter5.cut10.ngauss100.pt'
device = 'cpu'


###   SCRIPT   ###


hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

os.chdir(os.path.dirname(os.path.realpath(__file__)))
base_dir = '/home/alex/Dropbox/keith/projects/mbgdml-h2o-meoh-mecn'
dset_dir = os.path.join(base_dir, 'data/')
model_dir = os.path.join(base_dir, 'models/')

model_path = os.path.join(model_dir, model_path)

csv_data = [csv_headers]
for job in jobs:
    dset_path = job['dset_path']
    group_key = job['group_key']
    nbody_orders = job['nbody_orders']
    save_dir = job['save_dir']
    name = job['name']

    data_keys = {
        'z': f'{group_key}/atomic_numbers',
        'R': f'{group_key}/geometry',
        'E': f'{group_key}/{job["E_key"]}',
        'F': f'{group_key}/{job["F_key"]}',
    }

    csv_data.append([dset_path, group_key])
    save_dir = os.path.join(base_dir, save_dir)
    pset_path = os.path.join(save_dir, name)
    if pset_path[-4:] != '.npz':
        pset_path += '.npz'


    dset_path = os.path.join(dset_dir, dset_path)
    nbody_idxs = [i-1 for i in nbody_orders]

    # Preparing the data set.
    rfile = File(dset_path)
    dset = {
        'z': rfile.get(data_keys['z']),
        'R': rfile.get(data_keys['R']),
        'E': rfile.get(data_keys['E']),
        'F': -rfile.get(data_keys['F']),
        'entity_ids': rfile.get(f'{group_key}/entity_ids'),
        'comp_ids': rfile.get(f'{group_key}/comp_ids'),
        'theory': theory,
        'r_unit': r_unit,
        'e_unit': e_unit,
    }
    if dset['E'].ndim == 0:
        dset['E'] = np.array([dset['E']])
    
    if in_ev:
        dset['E'] *= hartree2ev
        dset['F'] *= hartree2ev
    else:
        dset['E'] *= hartree2kcalmol
        dset['F'] *= hartree2kcalmol

    os.makedirs(save_dir, exist_ok=True)

    # Checks to see if predict set already exists.
    if os.path.isfile(pset_path) and not overwrite:
        raise FileExistsError(f'{pset_path} already exists and overwrite is False.')

    t_start = time.time()

    print(f'Loading model')
    model = schnetModel(model_path, ['meoh'], device)

    

    print(f'\nMaking predictions')
    E_pred = np.empty(dset['E'].shape)
    F_pred = np.empty(dset['F'].shape)
    atom_conv = schnetpack.data.atoms.AtomsConverter(
        device=torch.device(model.device)
    )
    for i in range(len(E_pred)):
        pred = model.spk_model(
            atom_conv(ase.Atoms(dset['z'], dset['R'][i]))
        )
        E_pred[i] = pred['energy'].cpu().detach().numpy()[0][0]
        F_pred[i] = pred['forces'].cpu().detach().numpy()[0]
    t_stop = time.time()
    t_pred = t_stop - t_start

    if save:
        print(f'\nSaving the predict set')
        pset.save(pset.name, pset.asdict(), save_dir)

    print('\nResults')
    # Compute energy and force prediction errors
    E_errors = E_pred - dset['E']
    F_errors = F_pred.flatten() - dset['F'].flatten()
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
        for i in range(1, 4):
            if i in nbody_orders:
                csv_data[-1].append(True)
            else:
                csv_data[-1].append(False)
        csv_data[-1].extend(
            [
                'kcal/mol', 'Angstrom', E_mae, E_rmse, E_sse, E_max_error,
                F_mae, F_rmse, F_sse, F_max_error, t_pred
            ]
        )

if save:
    with open(os.path.join(base_dir, csv_dir, csv_name), 'w') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerows(csv_data)
