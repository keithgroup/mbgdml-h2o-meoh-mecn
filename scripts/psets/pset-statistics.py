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
Compute error statistics for predict sets.
"""

import os
import numpy as np
from mbgdml.data import predictSet


systems_dict = {
    'H2O mbGDML': {
        'pset_paths': [
            'h2o/gdml/4h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
            'h2o/gdml/5h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
            'h2o/gdml/6h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
        ],
        'pset_16mer_path': [
            'h2o/gdml/16h2o.yoo.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gdml.train1000.npz'
        ],
    },
    'H2O mbGAP': {
        'pset_paths': [
            'h2o/gap/4h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gap.train1000.npz',
            'h2o/gap/5h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gap.train1000.npz',
            'h2o/gap/6h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gap.train1000.npz',
        ],
        'pset_16mer_path': [
            'h2o/gap/16h2o.yoo.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gap.train1000.npz'
        ],
    },
    'H2O mbSchNet': {
        'pset_paths': [
            'h2o/schnet/4h2o.temelso.etal-pset-schnet.niter5.nfeat128.best.train1000.npz',
            'h2o/schnet/5h2o.temelso.etal-pset-schnet.niter5.nfeat128.best.train1000.npz',
            'h2o/schnet/6h2o.temelso.etal-pset-schnet.niter5.nfeat128.best.train1000.npz',
        ],
        'pset_16mer_path': [
            'h2o/schnet/16h2o.yoo.etal-pset-schnet.niter5.nfeat128.best.train1000.npz'
        ],
    },
    'MeCN mbGDML': {
        'pset_paths': [
            'mecn/gdml/4mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
            'mecn/gdml/5mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
            'mecn/gdml/6mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
        ],
        'pset_16mer_path': [
            'mecn/gdml/16mecn.remya.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gdml.train1000.npz'
        ],
    },
    'MeCN mbGAP': {
        'pset_paths': [
            'mecn/gap/4mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gap.train1000.npz',
            'mecn/gap/5mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gap.train1000.npz',
            'mecn/gap/6mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gap.train1000.npz',
        ],
        'pset_16mer_path': [
            'mecn/gap/16mecn.remya.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gap.train1000.npz'
        ],
    },
    'MeCN mbSchNet': {
        'pset_paths': [
            'mecn/schnet/4mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz',
            'mecn/schnet/5mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz',
            'mecn/schnet/6mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz',
        ],
        'pset_16mer_path': [
            'mecn/schnet/16mecn.remya.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz'
        ],
    },
    'MeOH mbGDML': {
        'pset_paths': [
            'meoh/gdml/4meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
            'meoh/gdml/5meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
            'meoh/gdml/6meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
        ],
        'pset_16mer_path': [
            'meoh/gdml/16meoh.pires.deturi-pset-62meoh.sphere.gfn2.md.500k.prod1-gdml.train1000.npz'
        ],
    },
    'MeOH mbGAP': {
        'pset_paths': [
            'meoh/gap/4meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-gap.train1000.npz',
            'meoh/gap/5meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-gap.train1000.npz',
            'meoh/gap/6meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-gap.train1000.npz',
        ],
        'pset_16mer_path': [
            'meoh/gap/16meoh.pires.deturi-pset-62meoh.sphere.gfn2.md.500k.prod1-gap.train1000.npz'
        ],
    },
    'MeOH mbSchNet': {
        'pset_paths': [
            'meoh/schnet/4meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz',
            'meoh/schnet/5meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz',
            'meoh/schnet/6meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz',
        ],
        'pset_16mer_path': [
            'meoh/schnet/16meoh.pires.deturi-pset-62meoh.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz'
        ],
    },
}

only_16mer = False  # False: errors of 4, 5, and 6mers


###   SCRIPT   ###
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
pset_dir = os.path.join(base_dir, 'data/psets/')

print('All energies in kcal/mol\n')
for model_key, pset_data in systems_dict.items():
    if only_16mer:
        pset_paths = pset_data['pset_16mer_path']
    else:
        pset_paths = pset_data['pset_paths']
    pset_paths = [os.path.join(pset_dir, path) for path in pset_paths]

    print(model_key)
    E_errors = []
    E_errors_per_monomer = []
    F_errors = []
    F_errors_per_atom = []
    for pset_path in pset_paths:
        pset = predictSet(pset_path, Z_key='z')
        n_monomers = int(len(set(pset.entity_ids)))
        n_atoms = int(len(pset.R[0]))

        E_true = pset.E_true
        F_true = pset.F_true

        E_pred, F_pred = pset.nbody_predictions([1, 2, 3])
        E_error = E_pred - E_true
        F_error = np.ravel(F_pred-F_true)
        if pset.e_unit.lower() == 'ev':
            E_error *= ev2kcalmol
            F_error *= ev2kcalmol
        F_error_per_atom = F_error/n_atoms
        E_error_per_monomer = E_error/n_monomers

        E_errors.extend(E_error.tolist())
        E_errors_per_monomer.extend(E_error_per_monomer.tolist())
        F_errors.extend(F_error.tolist())
        F_errors_per_atom.extend(F_error_per_atom.tolist())

    E_errors = np.array(E_errors)
    E_errors_per_monomer = np.array(E_errors_per_monomer)
    F_errors = np.array(F_errors)
    F_errors_per_atom = np.array(F_errors_per_atom)

    E_mae = np.mean(np.abs(E_errors))
    E_rmse = np.sqrt(np.mean((E_errors)**2))
    E_mae_per_monomer = np.mean(np.abs(E_errors_per_monomer))
    E_rmse_per_monomer = np.sqrt(np.mean((E_errors_per_monomer)**2))

    F_mae = np.mean(np.abs(F_errors))
    F_rmse = np.sqrt(np.mean((F_errors)**2))
    F_mae_per_atom = np.mean(np.abs(F_errors_per_atom))
    F_rmse_per_atom = np.sqrt(np.mean((F_errors_per_atom)**2))
    
    print('-----------------')
    print(f'Energy MAE/RMSE                 {E_mae:.3f} / {E_rmse:.3f}')
    print(f'Energy per monomer MAE/RMSE     {E_mae_per_monomer:.3f} / {E_rmse_per_monomer:.3f}')

    print(f'Force MAE/RMSE                  {F_mae:.3f} / {F_rmse:.3f}')
    print(f'Force per atom MAE/RMSE         {F_mae_per_atom:.3f} / {F_rmse_per_atom:.3f}')
    print()