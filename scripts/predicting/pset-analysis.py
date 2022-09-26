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

"""Print energy statistics from predict set"""

from mbgdml.data import predictSet
import numpy as np
import os

pset_path = 'meoh/gap/16meoh.pires.deturi-pset-62meoh.sphere.gfn2.md.500k.prod1-gap.train1000.npz'

# Conversion constants
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

# Setup paths
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))
base_dir = '../../'
pset_dir = 'data/psets/'
pset_path = os.path.join(base_dir, pset_dir, pset_path)

# Load in paths
pset = predictSet(pset_path)

# Check eV conversion.
e_unit = pset.e_unit
if e_unit.lower() == 'ev':
    e_conv = ev2kcalmol
else:
    e_conv = 1.0

E_true = pset.E_true * e_conv
F_true = pset.F_true * e_conv
E_pred, F_pred = pset.nbody_predictions([1, 2, 3])
E_pred *= e_conv
F_pred *= e_conv

E_error = E_pred - E_true
F_error = F_pred - F_true

E_mae = np.mean(np.abs(E_error))
F_mae = np.mean(np.abs(F_error.flatten()))
print(f'Energy MAE: {E_mae:.4f} kcal/mol')
print(f'Force MAE: {F_mae:.4f} kcal/(mol A)')
