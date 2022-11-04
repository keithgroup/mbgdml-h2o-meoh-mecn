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

"""Write schnetpack database from exdir format."""

import os
import numpy as np
from reptar import File
from reptar.writers import write_schnetpack_db
from mbgdml.descriptors import Criteria, com_distance_sum
from mbgdml.utils import get_entity_ids

# Reptar file to read
exdir_path = '62meoh-xtb.md-samples.exdir'

# Many-body dataset to write
save_path = 'meoh/schnetpack/62meoh.sphere.gfn2.md.500k.prod1-dset-all.db'

group_key = '3meoh'
energy_key = 'energy_ele_mp2.def2tzvp_orca'
grad_key = 'grads_mp2.def2tzvp_orca'

use_criteria = True
desc_arg_keys = ('atomic_numbers', 'geometry', 'entity_ids')
cutoff = 14.0





##################
###   Script   ###
##################

hartree2ev = 27.21138602  # Psi4 v1.5 constant

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
exdir_dir = os.path.join(base_dir, 'data/md-sampling/')
save_dir = os.path.join(base_dir, 'data/ml-dsets/')

exdir_path = os.path.join(exdir_dir, exdir_path)
save_path = os.path.join(save_dir, save_path)

print(f'Loading {exdir_path}')
rfile = File(exdir_path, mode='r')
Z = rfile.get(f'{group_key}/atomic_numbers')
R = rfile.get(f'{group_key}/geometry')
E = rfile.get(f'{group_key}/{energy_key}')  # Eh
E *= hartree2ev  # eV
G = rfile.get(f'{group_key}/{grad_key}')  # Eh/A
G *= hartree2ev  # eV/A
F = -G

if use_criteria:
    desc_kwargs = {'entity_ids': rfile.get(f'{group_key}/entity_ids')}
    r_criteria = Criteria(com_distance_sum, desc_kwargs, cutoff)
    R_idxs, _ = r_criteria.accept(Z, R)
    R = R[R_idxs]
    E = E[R_idxs]
    F = F[R_idxs]

print(f'Writing {save_path}')
db = write_schnetpack_db(
    save_path, Z, R, energy=E, forces=F
)

print('\nDone!')
