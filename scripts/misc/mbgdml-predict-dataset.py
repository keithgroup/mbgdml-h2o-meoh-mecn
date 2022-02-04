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
Predicts energies and forces of all structures in data set.
"""

import os
import time
import numpy as np
from mbgdml.data import dataSet
from mbgdml.predict import mbPredict

# Data set paths.
dset_dir = '../../data/datasets'
dset_dir_h2o = f'{dset_dir}/h2o'
dset_dir_mecn = f'{dset_dir}/mecn'
dset_dir_meoh = f'{dset_dir}/meoh'

# Model paths.
model_dir = '../../data/models'
model_dir_h2o = f'{model_dir}/h2o'
model_dir_mecn = f'{model_dir}/mecn'
model_dir_meoh = f'{model_dir}/meoh'

solvent = 'mecn'  # h2o, mecn, meoh
dset_dir_solvent = dset_dir_mecn
model_dir_solvent = model_dir_mecn

dataset_path = f'{dset_dir_solvent}/6mecn/6mecn.malloum.etal-dset.npz'
model_paths = [
    f'{model_dir_solvent}/1{solvent}/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain1000.npz',
    f'{model_dir_solvent}/2{solvent}/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain1000.npz',
    f'{model_dir_solvent}/3{solvent}/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain1000.npz',
]

ignore_criteria = False
use_torch = False
time_it = False  # Will take a average of num_pred predictions
num_pred = 10
profile = False

all_same_entities = True



all_force_error = []


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if solvent == 'h2o':
    atoms_per_mol = 3
elif solvent == 'mecn' or solvent == 'meoh':
    atoms_per_mol = 6
else:
    raise ValueError(f'{solvent} is not a valid option.')

if not all_same_entities:
    raise ValueError(f'Only works for structures with all the same chemical species.')

def get_entity_ids(atoms_per_mol, num_mol):
    """Prepares the list of entity ids for a system with only one species.

    Note that all of the atoms in each molecule must occur in the same order and
    be grouped together.

    Parameters
    ----------
    atoms_per_mol : :obj:`int`
        Number of atoms in the molecule.
    num_mol : :obj:`int`
        Number of molecules in the system.
    """
    entity_ids = []
    for i in range(0, num_mol):
        entity_ids.extend([i for _ in range(0, atoms_per_mol)])
    return np.array(entity_ids)

def get_comp_ids(solvent, entity_ids):
    """Prepares the list of component ids for a system with only one species.

    Parameters
    ----------
    atoms_per_mol : :obj:`int`
        Number of atoms in the molecule.
    num_mol : :obj:`int`
        Number of molecules in the system.
    
    Returns
    -------
    :obj:`numpy.ndarray`
    """
    entity_ids_set = set(entity_ids)
    comp_ids = []
    for i in entity_ids_set:
        comp_ids.append([i, solvent])
    return np.array(comp_ids)

def main():
    print(f'Loading data set and models')
    dset = dataSet(dataset_path)
    predict = mbPredict(model_paths, use_torch=use_torch)

    num_entities = int(len(set(dset.entity_ids)))
    entity_ids = get_entity_ids(atoms_per_mol, num_entities)
    comp_ids = get_comp_ids(solvent, entity_ids)
    
    print(f'Predicting {len(dset.R)} structures')
    print('---------------------------------')
    for i in range(len(dset.R)):
        print(f'Structure {i}')
        r = dset.R[i]
        e_true = dset.E[i]
        f_true = dset.F[i]
        if time_it:
            predict_time = []
            for i in range(num_pred):
                start_t = time.time()
                e, f = predict.predict(
                    dset.z, r, entity_ids, comp_ids,
                    ignore_criteria=ignore_criteria
                )
                end_t = time.time()
                predict_time.append(end_t - start_t)
            predict_time = np.average(np.array(predict_time))
            print(f'Avg. time: {predict_time:.3f} seconds')
        else:
            e, f = predict.predict(
                dset.z, r, entity_ids, comp_ids,
                ignore_criteria=ignore_criteria
            )
        e_error = e - e_true
        f_error = f - f_true
        all_force_error.append(f_error.flatten())
        f_mse = np.square(np.subtract(f_true, f)).mean()
        f_rmse = np.sqrt(f_mse)
        """
        print(f'True energy:   {e_true:.1f} kcal/mol')
        print(f'Pred. energy:  {e[0]:.1f} kcal/mol')
        print(f'Energy error:        {e_error[0]:.1f} kcal/mol')
        print(f'\nForce RMSE:   {f_rmse:.3f} kcal/(mol A)')
        print('---------------------------------')
        """

        # DEBUG
        #print(e)
        #print(f)


if profile:
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats()
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(50)
else:
    main()

all_force_error = np.array(all_force_error)
print(np.mean(np.abs(all_force_error.flatten())))