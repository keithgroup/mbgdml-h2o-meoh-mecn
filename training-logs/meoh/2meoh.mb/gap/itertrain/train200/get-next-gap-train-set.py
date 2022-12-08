from mbgdml.analysis.problematic import ProblematicStructures
from mbgdml.descriptors import Criteria, com_distance_sum
from mbgdml.utils import get_entity_ids
from mbgdml.data import DataSet
from mbgdml.models import gapModel
from mbgdml.predictors import predict_gap
import numpy as np
import os

import time
t_start = time.time()

dset_path = '62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh-dset.mb-cm8.npz'
model_path = '2meoh.mb-gap.xml'
train_idxs_path = 'train-idxs-200.npy'
next_train_idxs_path = 'train-idxs-300.npy'


###   SCRIPT   ###
base_dir = '/ihome/jkeith/amm503/projects/gap-training-2022-06/training-logs/meoh/2meoh/train200'
dset_path = os.path.join(base_dir, dset_path)
model_path = os.path.join(base_dir, model_path)
train_idxs_path = os.path.join(base_dir, train_idxs_path)
next_train_idxs_path = os.path.join(base_dir, next_train_idxs_path)

dset = DataSet(dset_path, Z_key='z')

desc_kwargs = {
    'entity_ids': get_entity_ids(atoms_per_mol=6, num_mol=2)  # 2meoh
}
mbe_cutoff = 8.0
model_comp_ids = ['meoh', 'meoh']
r_criteria = Criteria(com_distance_sum, desc_kwargs, mbe_cutoff)
model = gapModel(model_path, model_comp_ids, r_criteria)

train_idxs = np.load(train_idxs_path)

# Find problematic structures
prob_s = ProblematicStructures([model], predict_gap)
prob_idxs = prob_s.find(dset, 100, dset_is_train=True, train_idxs=train_idxs)

next_train_idxs = np.concatenate((train_idxs, prob_idxs))

np.save(next_train_idxs_path, next_train_idxs)

t_end = time.time()
print(f'Elapsed time: {t_end-t_start} s')


base_dir = '/ihome/jkeith/amm503/projects/gap-training-2022-06/training-logs/meoh/2meoh/train200'

# Writing next training set
from reptar import File
from reptar.writers import write_xyz_gap
from reptar.utils import find_parent_r_idxs


# Reptar file to read
exdir_path = '/ihome/jkeith/amm503/projects/gap-training-2022-06/training-logs/meoh/2meoh/62meoh-xtb.md-samples.exdir'

# Many-body dataset to write.
# train, valid, and test labels will be automatically added.
save_path = os.path.join(base_dir, 'train-300.xyz')

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
    R_idxs, _ = r_criteria.accept(Z, R)
    R = R[R_idxs]
    E = E[R_idxs]
    F = F[R_idxs]

print(f'Writing {save_path}')

if use_idxs:
    write_xyz_gap(save_path, lattice, Z, R[next_train_idxs], E[next_train_idxs], F=F[next_train_idxs])
else:
    write_xyz_gap(save_path, lattice, Z, R, E)

print('\nDone!')
