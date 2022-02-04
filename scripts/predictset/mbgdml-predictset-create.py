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
Creates a prediction set.
"""

import os
from mbgdml.data import predictSet

# Data set dirs.
dataset_dir = '../../data/datasets'
dataset_dir_h2o = f'{dataset_dir}/h2o'
dataset_dir_mecn = f'{dataset_dir}/mecn'
dataset_dir_meoh = f'{dataset_dir}/meoh'

# Model dirs.
model_dir = '../../data/models'
model_dir_h2o = f'{model_dir}/h2o'
model_dir_mecn = f'{model_dir}/mecn'
model_dir_meoh = f'{model_dir}/meoh'

# Predict set dirs
predictset_dir = '../../data/predictsets'
predictset_dir_h2o = f'{predictset_dir}/h2o'
predictset_dir_mecn = f'{predictset_dir}/mecn'
predictset_dir_meoh = f'{predictset_dir}/meoh'

dataset_dir_solvent = dataset_dir_mecn
model_dir_solvent = model_dir_mecn
predictset_dir_solvent = predictset_dir_mecn

dataset_path = f'{dataset_dir_solvent}/6mecn/6mecn.malloum.etal.1.md.orca.mp2.def2tzvp.300k-dset.npz'

"""
# h2o models
model_paths = [
    f'{model_dir_solvent}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain1000.npz',
    f'{model_dir_solvent}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6-model.mb-iterativetrain1000.npz',
    f'{model_dir_solvent}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain1000.npz',
]
"""
# mecn models
model_paths = [
    f'{model_dir_solvent}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain1000.npz',
    f'{model_dir_solvent}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain1000.npz',
    f'{model_dir_solvent}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain1000.npz',
]
"""
# meoh models
model_paths = [
    f'{model_dir_solvent}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-iterativetrain1000.npz',
    f'{model_dir_solvent}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain1000.npz',
    f'{model_dir_solvent}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain1000.npz',
]
"""

predictset_name = '6mecn.malloum.etal.1.md.orca.mp2.def2tzvp.300k-pset-48mecn.sphere.gfn2.md.500k.prod1.iterativetrain1000'
save_dir = f'{predictset_dir_solvent}/6mecn/'
overwrite = False
save = True





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if save_dir[-1] != '/':
    save_dir += '/'

def main():

    os.makedirs(save_dir, exist_ok=True)

    # Checks to see if predict set already exists.
    if os.path.isfile(f'{save_dir}{predictset_name}.npz') and not overwrite:
        print(f'{save_dir}{predictset_name}.npz already exists and overwrite is False.\n')
        raise FileExistsError

    # Creating the predict set.
    pset = predictSet()

    print(f'Loading the {dataset_path} data set')
    pset.load_dataset(dataset_path)

    print(f'Loading {len(model_paths)} model(s)')
    pset.load_models(model_paths)

    print(f'Creating the predict set')
    pset.name = predictset_name
    pset.prepare(pset.z, pset.R, pset.entity_ids, pset.comp_ids)
    if save:
        print(f'Saving the predict set')
        pset.save(predictset_name, pset.asdict, save_dir)

    # DEBUG


main()
