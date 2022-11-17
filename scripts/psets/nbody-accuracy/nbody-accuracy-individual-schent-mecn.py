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

"""Determines origin of energy errors (cutoff or prediction)."""

from mbgdml.data import predictSet
import numpy as np
import os


pset_path = 'mecn/3mecn/schnet/16mecn.remya.etal.3mecn-pset-48mecn.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz'
nbody_order = 3
in_ev = True  # Is the pset in eV? If False we assume kcal/mol
print_n_errors = 10


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
data_dir = os.path.join(base_dir, 'data/')
pset_dir = os.path.join(data_dir, 'psets/')
pset_path = os.path.join(pset_dir, pset_path)

hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

pset = predictSet(pset_path, Z_key='z')
E_pred, F_pred = pset.nbody_predictions([nbody_order])
E_true = pset.E_true
F_true = pset.F_true

if in_ev:
    E_pred *= ev2kcalmol
    F_pred *= ev2kcalmol
    E_true *= ev2kcalmol
    F_true *= ev2kcalmol

E_error = E_pred - E_true
E_error = np.nan_to_num(E_error, nan=0.0)
worst_idxs = np.argsort(np.abs(E_error))[::-1]

print('Largest energy errors')
print('---------------------')
for i in worst_idxs[:print_n_errors]:
    print(f'Energy error: {E_error[i]:.3f} kcal/mol')
    