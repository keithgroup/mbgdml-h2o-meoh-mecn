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

"""Compute energy and force errors statistics for (MeOH)20 with SchNet.
"""

import os
import numpy as np
from reptar import File
from mbgdml.mbe import mbePredict
from mbgdml.models import schnetModel
from mbgdml.predictors import predict_schnet
from mbgdml.descriptors import Criteria, com_distance_sum
from mbgdml.utils import get_entity_ids

model_paths = [
    'meoh/1meoh/schnet/1meoh-niter5.nfeat128.cut5.ngauss50-train1000.pt',
    'meoh/2meoh/schnet/2meoh.mb-niter5.nfeat128.cut10-train1000.pt',
    'meoh/3meoh/schnet/3meoh.mb-niter5.nfeat128.cut10-train1000.pt',
]
in_ev = True  # GAP and SchNet models are in eV

# L cutoffs
# Solvent | 2-body | 3-body |
# H2O     |   6    |   10   |
# MeCN    |   9    |   17   |
# MeOH    |   8    |   14   |
model_desc_cutoffs = (None, 8.0, 14.0)
models_comp_ids = [['meoh'], ['meoh', 'meoh'], ['meoh', 'meoh', 'meoh']]
model_desc_kwargs = (
    {'entity_ids': get_entity_ids(atoms_per_mol=6, num_mol=1)},  # 1meoh
    {'entity_ids': get_entity_ids(atoms_per_mol=6, num_mol=2)},  # 2meoh
    {'entity_ids': get_entity_ids(atoms_per_mol=6, num_mol=3)},  # 3meoh
)
model_criterias = [
    Criteria(com_distance_sum, desc_kwargs, cutoff) for desc_kwargs,cutoff \
    in zip(model_desc_kwargs, model_desc_cutoffs)
]
all_data_keys = {
    'yao': {
        'rfile_path': 'isomers/20meoh.yao.etal.exdir',
        'group_key': '/',
        'energy_key': 'energy_ele_rimp2_def2tzvp',
    },
}

yao_isomer_order = [3, 2, 4, 1]
yao_isomer_energies = np.array([30.961317, 40.888517, 50.840818, 52.007985])

E_ref_ccpvtz = np.array([30.77266394, 40.26531391, 50.24385102, 51.6564622])
E_ref_def2tzvp = np.array([28.5425054, 38.8793339, 49.94528484, 49.51020351])


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
model_paths = [os.path.join(model_dir, model_path) for model_path in model_paths]

# Setup mbML predictor
# Initialize model
models = []
for i in range(len(model_paths)):
    models.append(
        schnetModel(
            model_paths[i], model_comp_ids, 'cpu', criteria=model_criterias[i]
        )
    )
mbpred = mbePredict(models, predict_schnet)

for key in all_data_keys:
    data_keys = all_data_keys[key]

    rfile_path = data_keys['rfile_path']
    rfile_path = os.path.join(data_dir, rfile_path)
    group_key = data_keys['group_key']
    energy_key = data_keys['energy_key']

    # Load in paths
    rfile = File(rfile_path, mode='r')
    # Get reference data
    Z_ref = rfile.get(f'{group_key}/atomic_numbers')
    R_ref = rfile.get(f'{group_key}/geometry')
    E_ref = rfile.get(f'{group_key}/{energy_key}')  # Eh

    # Convert units
    E_ref *= hartree2kcalmol  # kcal/mol

    entity_ids = rfile.get(f'{group_key}/entity_ids')
    comp_ids = rfile.get(f'{group_key}/comp_ids')

    E_pred, F_pred = mbpred.predict(Z_ref, R_ref, entity_ids, comp_ids)
    if in_ev:
        E_pred *= ev2kcalmol

    E_rel = E_pred - E_pred[0]
    E_rel = E_rel[yao_isomer_order]
    print(f'True        : {E_ref_def2tzvp}')
    print(f'Predictions : {E_rel}')
    print(f'Differences : {E_rel-E_ref_def2tzvp}')

