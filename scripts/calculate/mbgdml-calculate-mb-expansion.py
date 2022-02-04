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

"""
Calculates many-body predictions using data sets and compares to supersystem
calculation.
"""

import os
import numpy as np
from mbgdml.data import dataSet
from mbgdml.utils import e_f_contribution

dset_dir = '../../data/datasets'
dset_dir_h2o = f'{dset_dir}/h2o'
dset_dir_mecn = f'{dset_dir}/mecn'
dset_dir_meoh = f'{dset_dir}/meoh'

dset_dir_solvent = dset_dir_mecn

structure_dset_path = f'{dset_dir_solvent}/6mecn/6mecn.malloum.etal-dset.npz'

nbody_dset_paths = [
    f'{dset_dir_solvent}/1mecn/malloum.etal/6mecn.malloum.etal.dset.1mecn-dset.npz',
    f'{dset_dir_solvent}/2mecn/malloum.etal/6mecn.malloum.etal.dset.2mecn-dset.mb.npz',
    f'{dset_dir_solvent}/3mecn/malloum.etal/6mecn.malloum.etal.dset.3mecn-dset.mb.npz',
]





###   SCRIPT   ###

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

structure_dset = dataSet(structure_dset_path)
nbody_dsets = [dataSet(dset_path) for dset_path in nbody_dset_paths]

# Prepares many-body data set by making all energies and forces zero.
structure_dset_mb = dataSet(structure_dset_path)
structure_dset_mb.E = np.zeros(structure_dset_mb.E.shape)
structure_dset_mb.F = np.zeros(structure_dset_mb.F.shape)

structure_dset_mb = e_f_contribution(structure_dset_mb, nbody_dsets, 'add')

E_mb = structure_dset_mb.E
F_mb = structure_dset_mb.F

E_true = structure_dset.E
E_error = E_mb - E_true

F_true = structure_dset.F

# Calculates F_rmse
F_error_all = F_mb - F_true
F_mae = np.zeros(E_error.shape)
F_rmse = np.zeros(E_error.shape)
for i in range(len(F_rmse)):
    F_mae[i] = np.mean(np.abs(np.subtract(F_true[i], F_mb[i])))
    f_mse = np.square(np.subtract(F_true[i], F_mb[i])).mean()
    F_rmse[i] = np.sqrt(f_mse)

num_R = len(structure_dset.R)

for i in range(num_R):
    print('---------------------------------')
    print(f'Structure {i+1}/{num_R}\n')
    print(f'True energy:   {E_true[i]:.1f} kcal/mol')
    print(f'MBE energy:    {E_mb[i]:.1f} kcal/mol')
    print(f'Energy error:        {E_error[i]:.1f} kcal/mol')
    print(f'\nForce MAE:     {F_mae[i]:.3f} kcal/(mol A)')
    print(f'Force RMSE:    {F_rmse[i]:.3f} kcal/(mol A)')
    print('---------------------------------')

