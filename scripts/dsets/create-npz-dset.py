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

"""Create NPZ data set containing just Z, R, E, and F"""

import os
import numpy as np
from reptar import File

exdir_path = 'data/md-sampling/140h2o-xtb.md-samples.exdir'

group_key = '3h2o'
e_key = 'energy_ele_mp2.def2tzvp_orca'
g_key = 'grads_mp2.def2tzvp_orca'


npz_path = 'data/ml-dsets/h2o/3h2o/3h2o.npz'






###   SCRIPT   ###
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
exdir_path = os.path.join(base_dir, exdir_path)
npz_path = os.path.join(base_dir, npz_path)



rfile = File(exdir_path, mode='r')

Z = rfile.get(f'{group_key}/atomic_numbers')
R = rfile.get(f'{group_key}/geometry')
E = rfile.get(f'{group_key}/{e_key}')  # Eh
F = -rfile.get(f'{group_key}/{g_key}')  # Eh/A

E *= hartree2kcalmol  # kcal/mol
F *= hartree2kcalmol  # kcal/(mol A)

npz_data = {
    'Z': Z, 'R': R, 'r_unit': 'Angstrom', 'E': E, 'e_unit': 'kcal/mol', 'F': F
}
np.savez(npz_path, **npz_data)
