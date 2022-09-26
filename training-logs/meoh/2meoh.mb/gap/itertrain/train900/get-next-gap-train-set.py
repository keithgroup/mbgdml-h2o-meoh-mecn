from mbgdml.analysis.problematic import prob_structures
from mbgdml.criteria import cm_distance_sum
from mbgdml.data import dataSet
from mbgdml.predict import gapModel, predict_gap
import numpy as np
import os

import time
t_start = time.time()

dset_path = '62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh-dset.mb-cm8.npz'
model_path = '2meoh.mb-gap.xml'
current_size = 900
next_size = 1000

train_idxs_path = f'train-idxs-{current_size}.npy'
next_train_idxs_path = f'train-idxs-{next_size}.npy'


###   SCRIPT   ###
base_dir = f'/ihome/jkeith/amm503/projects/gap-training-2022-06/training-logs/meoh/2meoh/train{current_size}'
dset_path = os.path.join(base_dir, dset_path)
model_path = os.path.join(base_dir, model_path)
train_idxs_path = os.path.join(base_dir, train_idxs_path)
next_train_idxs_path = os.path.join(base_dir, next_train_idxs_path)

dset = dataSet(dset_path)

mbe_cutoff = 8.0
model_comp_ids = ['meoh', 'meoh']
model = gapModel(
    model_path, model_comp_ids,
    criteria_desc_func=cm_distance_sum, criteria_cutoff=mbe_cutoff
)

train_idxs = np.load(train_idxs_path)

# Find problematic structures
prob_s = prob_structures([model], predict_gap)
prob_idxs = prob_s.find(dset, 100, dset_is_train=True, train_idxs=train_idxs)

next_train_idxs = np.concatenate((train_idxs, prob_idxs))
np.save(next_train_idxs_path, next_train_idxs)

t_end = time.time()
print(f'Elapsed time: {t_end-t_start} s')



# Writing next training set
from reptar import File
from reptar.writers import write_xyz_gap
from reptar.descriptors import criteria, com_distance_sum
from reptar.utils import find_parent_r_idxs


# Reptar file to read
exdir_path = '/ihome/jkeith/amm503/projects/gap-training-2022-06/training-logs/meoh/2meoh/62meoh-xtb.md-samples.exdir'

# Many-body dataset to write.
# train, valid, and test labels will be automatically added.
save_path = os.path.join(base_dir, f'train-{next_size}.xyz')

# Reference GDML model (for training and validation indices).
# These are applied after any criteria.
use_idxs = True

group_key = '2meoh'
energy_key = 'energy_ele_nbody_mp2.def2tzvp_orca'
grad_key = 'grads_nbody_mp2.def2tzvp_orca'

lattice = np.array(
    [[200.0,   0.0,   0.0],
     [  0.0, 200.0,   0.0],
     [  0.0,   0.0, 200.0]]
)

use_criteria = True
desc_arg_keys = ('atomic_numbers', 'geometry', 'entity_ids')
mbe_cutoff = 8.0




##################
###   Script   ###
##################

hartree2ev = 27.21138602  # Psi4 constant


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
    desc_args = (rfile.get(f'{group_key}/{dkey}') for dkey in desc_arg_keys)
    _, R_idxs = criteria(com_distance_sum, desc_args, mbe_cutoff)
    R = R[R_idxs]
    E = E[R_idxs]
    F = F[R_idxs]

print(f'Writing {save_path}')

if use_idxs:
    write_xyz_gap(save_path, lattice, Z, R[next_train_idxs], E[next_train_idxs], F=F[next_train_idxs])
else:
    write_xyz_gap(save_path, lattice, Z, R, E)

print('\nDone!')
