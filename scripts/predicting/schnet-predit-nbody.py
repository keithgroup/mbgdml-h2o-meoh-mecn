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

"""Use a GAP model to predict n-body energies and forces with a reference"""

import torch
import os
import numpy as np
from reptar import File
from reptar.writers import write_schnetpack_db
import schnetpack as spk

struct_path = 'isomers/h2o-temelso.etal.exdir'

ref_group_key = '6h2o/samples_3h2o'  # Key to total energies and forces
nbody_group_keys = '6h2o/samples_3h2o'  # Keys to all n-body groups for reference

E_key = 'energy_ele_mp2.def2tzvp_orca'
G_key = 'grads_mp2.def2tzvp_orca'
E_mb_key = 'energy_ele_nbody_mp2.def2tzvp_orca'
G_mb_key = 'grads_nbody_mp2.def2tzvp_orca'

E_ref_key = E_mb_key  # Reference energies to compute errors to
G_ref_key = G_mb_key  # Reference gradients to compute errors to

model_path = 'h2o/3h2o/schnet/3h2o.mb-niter5.nfeat128.cut10-train1000.pt'
device = 'cpu'


##################
###   SCRIPT   ###
##################
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant

# Setup paths
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
model_dir = '../../../mbgdml-h2o-meoh-mecn-models'
struct_dir = os.path.join(base_dir, 'data')
model_path = os.path.join(model_dir, model_path)
struct_path = os.path.join(struct_dir, struct_path)

# Load in paths
rfile = File(struct_path)

# Initialize model
schnet = torch.load(model_path, map_location=device)


# Get reference data
Z_ref = rfile.get(f'{ref_group_key}/atomic_numbers')
R_ref = rfile.get(f'{ref_group_key}/geometry')
E_ref = np.array(rfile.get(f'{ref_group_key}/{E_ref_key}'))  # Eh
G_ref = np.array(rfile.get(f'{ref_group_key}/{G_ref_key}'))  # Eh/A

# Convert units
E_ref *= hartree2ev  # eV
G_ref *= hartree2ev  # eV/A
F_ref = np.negative(G_ref)

tmp_db_path = '/tmp/test.db'
if os.path.exists(tmp_db_path):
    os.remove(tmp_db_path)
db = write_schnetpack_db(
    tmp_db_path, Z_ref, R_ref, energy=E_ref, forces=F_ref
)
loader = spk.AtomsLoader(db)

E_preds = []
F_preds = []
for count, batch in enumerate(loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    pred = schnet(batch)
    
    E_preds.extend(pred['energy'].cpu().detach().numpy()[:,0])
    F_preds.extend(pred['forces'].cpu().detach().numpy())

E_preds = np.array(E_preds)
F_preds = np.array(F_preds)

E_errors = E_preds - E_ref
F_errors = F_preds - F_ref

E_mae = np.mean(np.abs(E_errors))
E_mae_kcalmol = E_mae / hartree2ev * hartree2kcalmol
F_mae = np.mean(np.abs(F_errors.flatten()))
F_mae_kcalmol = F_mae / hartree2ev * hartree2kcalmol
print('Summary')
print('-------')
print(f'Energy MAE: {E_mae:.4f} eV    =  {E_mae_kcalmol:.4f} kcal/mol')
print(f'Force MAE:  {F_mae:.4f} eV/A  =  {F_mae_kcalmol:.4f} kcal/(mol A)')
