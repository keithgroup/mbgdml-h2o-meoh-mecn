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

"""Compute energy and force error statistics for 3-body interactions on
4-6mer water isomers with GAP.
"""

import os
import numpy as np
from reptar import File
from mbgdml.mbe import mbePredict
from mbgdml.models import gapModel
from mbgdml.predictors import predict_gap
from mbgdml.criteria import cm_distance_sum

model_path = 'h2o/3h2o/gap/3h2o.mb-gap.xml'
in_ev = True  # GAP and SchNet models are in eV
model_comp_ids = ['h2o', 'h2o', 'h2o']

# L cutoffs
# Solvent | 2-body | 3-body |
# H2O     |   6    |   10   |
# MeCN    |   9    |   17   |
# MeOH    |   8    |   14   |
mb_cutoff = 10.0  # Water 3-body L cutoff
all_data_keys = {
    '4h2o': {
        'rfile_path': 'isomers/h2o-temelso.etal.exdir',
        'group_key': '4h2o/samples_3h2o',
        'energy_key': 'energy_ele_nbody_mp2.def2tzvp_orca',
        'gradient_key': 'grads_nbody_mp2.def2tzvp_orca',
    },
    '5h2o': {
        'rfile_path': 'isomers/h2o-temelso.etal.exdir',
        'group_key': '5h2o/samples_3h2o',
        'energy_key': 'energy_ele_nbody_mp2.def2tzvp_orca',
        'gradient_key': 'grads_nbody_mp2.def2tzvp_orca',
    },
    '6h2o': {
        'rfile_path': 'isomers/h2o-temelso.etal.exdir',
        'group_key': '6h2o/samples_3h2o',
        'energy_key': 'energy_ele_nbody_mp2.def2tzvp_orca',
        'gradient_key': 'grads_nbody_mp2.def2tzvp_orca',
    },
}


##################
###   SCRIPT   ###
##################
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

# Setup paths
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
model_dir = '../../../../mbgdml-h2o-meoh-mecn-models'
data_dir = os.path.join(base_dir, 'data')
model_path = os.path.join(model_dir, model_path)

# Setup mbML predictor
# Initialize model
models = [
    gapModel(
        model_path, model_comp_ids, criteria_desc_func=cm_distance_sum,
        criteria_cutoff=mb_cutoff
    )
]
mbpred = mbePredict(models, predict_gap)

for key in all_data_keys:
    data_keys = all_data_keys[key]

    rfile_path = data_keys['rfile_path']
    rfile_path = os.path.join(data_dir, rfile_path)
    group_key = data_keys['group_key']
    energy_key = data_keys['energy_key']
    gradient_key = data_keys['gradient_key']

    # Load in paths
    rfile = File(rfile_path, mode='r')
    # Get reference data
    Z_ref = rfile.get(f'{group_key}/atomic_numbers')
    R_ref = rfile.get(f'{group_key}/geometry')
    E_ref = rfile.get(f'{group_key}/{energy_key}')  # Eh
    G_ref = rfile.get(f'{group_key}/{gradient_key}')  # Eh/A

    # Convert units
    E_ref *= hartree2kcalmol  # kcal/mol
    G_ref *= hartree2kcalmol  # kcal/(mol A)
    F_ref = np.negative(G_ref)

    entity_ids = rfile.get(f'{group_key}/entity_ids')
    comp_ids = rfile.get(f'{group_key}/comp_ids')

    E_pred, F_pred = mbpred.predict(Z_ref, R_ref, entity_ids, comp_ids)
    if in_ev:
        E_pred *= ev2kcalmol
        F_pred *= ev2kcalmol

    E_errors = E_pred - E_ref
    F_errors = F_pred - F_ref
    F_errors = F_errors.flatten()


    E_mae = np.mean(np.abs(E_errors))
    E_rmse = np.sqrt(np.mean((E_errors)**2))
    F_mae = np.mean(np.abs(F_errors))
    F_rmse = np.sqrt(np.mean((F_errors)**2))

    print(f'Key: {key}')
    print(f'Energy MAE / RMSE :   {E_mae:.3f} / {E_rmse:.3f}   kcal/mol')
    print(f'Force  MAE / RMSE :   {F_mae:.3f} / {F_rmse:.3f}   kcal/(mol A)')
    print()
