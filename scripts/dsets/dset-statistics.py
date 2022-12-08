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
Prints statistics like min, mean, max, and standard deviation of energy and 
forces of data sets.
"""

import os
import numpy as np
from mbgdml.data import DataSet

dset_path = 'meoh/3meoh/gdml/62meoh.sphere.gfn2.md.500k.prod1.3meoh-dset.mb-cm14.npz'


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../data/ml-dsets'
dset_path = os.path.join(base_dir, dset_path)

dset = DataSet(dset_path, Z_key='z')

E = dset.E
F = dset.F

# Energy statistics
E_min = np.min(E)
E_mean = np.mean(E)
E_max = np.max(E)
E_stdev = np.std(E)

# Force statistics
F_min = np.min(F.flatten())
F_mean = np.mean(F.flatten())
F_max = np.max(F.flatten())
F_stdev = np.std(F.flatten())


print(f'Energy statistics')
print(f'Min:   {E_min:.3f} kcal/mol')
print(f'Mean:  {E_mean:.3f} kcal/mol')
print(f'Max:   {E_max:.3f} kcal/mol')
print(f'Stdev: {E_stdev:.3f} kcal/mol')

print(f'\nForce statistics')
print(f'Min:   {F_min:.3f} kcal/mol')
print(f'Mean:  {F_mean:.3e} kcal/mol')
print(f'Max:   {F_max:.3f} kcal/mol')
print(f'Stdev: {F_stdev:.3f} kcal/mol')

