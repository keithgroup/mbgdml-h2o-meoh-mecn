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

"""Create a GAP dataset from a reptar file and mbGDML dataset.
Must split into train, validation, and test data sets, so we take these splits
from mbGDML training."""

import os
import numpy as np
from reptar import File
from reptar.writers import write_xyz_gap
from mbgdml.descriptors import Criteria, com_distance_sum
from mbgdml.utils import get_entity_ids
from reptar.utils import find_parent_r_idxs

# Reptar file to read
exdir_path = 'md-sampling/62meoh-xtb.md-samples.exdir'

# Many-body dataset to write.
# train, valid, and test labels will be automatically added.
save_path = 'meoh/2meoh/gap/iterative-train/200/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh-dset.mb-cm8.xyz'

# Reference GDML model (for training and validation indices).
# These are applied after any criteria.
use_idxs = True
train_idxs_path = 'meoh/2meoh.mb/gdml/train200/train_idxs.npy'
valid_idxs_path = 'meoh/2meoh.mb/gdml/train200/valid_idxs.npy'
test_idxs_path = 'meoh/2meoh.mb/gdml/train200/test_idxs.npy'

group_key = '2meoh'
energy_key = 'energy_ele_nbody_mp2.def2tzvp_orca'
grad_key = 'grads_nbody_mp2.def2tzvp_orca'

lattice = np.array(
    [[200.0,   0.0,   0.0],
     [  0.0, 200.0,   0.0],
     [  0.0,   0.0, 200.0]]
)

use_criteria = True
cutoff = 8.0





##################
###   Script   ###
##################

hartree2ev = 27.21138602  # Psi4 constant

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

dir_base = '../../'
dir_exdir = os.path.join(dir_base, 'data/')
dir_save = os.path.join(dir_base, 'data/ml-dsets/')
dir_log = os.path.join(dir_base, 'training-logs/')

exdir_path = os.path.join(dir_exdir, exdir_path)
save_path = os.path.join(dir_save, save_path)
train_idxs_path = os.path.join(dir_log, train_idxs_path)
valid_idxs_path = os.path.join(dir_log, valid_idxs_path)
test_idxs_path = os.path.join(dir_log, test_idxs_path)


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
    desc_kwargs = {
        'entity_ids': rfile.get(f'{group_key}/entity_ids')
    }
    r_criteria = Criteria(com_distance_sum, desc_kwargs, cutoff)
    R_idxs, _ = r_criteria.accept(Z, R)
    R = R[R_idxs]
    E = E[R_idxs]
    F = F[R_idxs]


train_idxs = np.load(train_idxs_path)
valid_idxs = np.load(valid_idxs_path)
test_idxs = np.load(test_idxs_path)


print(f'Writing {save_path}')
if use_idxs:
    for xyz_label,idxs in zip(
        ('-train', '-valid', '-test'),
        (train_idxs, valid_idxs, test_idxs)
    ):
        xyz_path_set = save_path[:-4] + xyz_label + save_path[-4:]
        write_xyz_gap(xyz_path_set, lattice, Z, R[idxs], E[idxs], F=F[idxs])
else:
    write_xyz_gap(save_path, lattice, Z, R, E)

print('\nDone!')
