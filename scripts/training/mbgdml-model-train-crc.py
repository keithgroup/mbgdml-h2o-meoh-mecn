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

import os
import numpy as np
from mbgdml.data import dataSet
from mbgdml.train import mbGDMLTrain

##### Job Information #####

dataset_dir = '/ihome/jkeith/amm503/projects/mbgdml/training'
dataset_dir_h2o = f'{dataset_dir}/h2o'
dataset_dir_mecn = f'{dataset_dir}/mecn'
dataset_dir_meoh = f'{dataset_dir}/meoh'

model_save_dir = '/ihome/jkeith/amm503/projects/mbgdml/training'
model_save_dir_h2o = f'{model_save_dir}/h2o'
model_save_dir_mecn = f'{model_save_dir}/mecn'
model_save_dir_meoh = f'{model_save_dir}/meoh'

dataset_dir_solvent = dataset_dir_h2o
model_save_dir_solvent = model_save_dir_h2o

dataset_path = f'{dataset_dir_solvent}/3h2o.mb-training/140h2o.sphere.gfn2.md.500k.prod1.3h2o-dset.mb-cm10.npz'
save_dir = f'{model_save_dir_solvent}/3h2o.mb-training'

dset = dataSet(dataset_path)
total_number = dset.n_R  # Number to total structures in data set.
num_train = 1000  # Number of data points to train on.
num_validate = 2000  # Number of data points to compare candidate models with.
num_test = total_number - num_train - num_validate  # Number of data points to test final model with.
if num_test > 3000:
    num_test = 3000
sigmas = list(range(300, 702, 5))  # Sigmas to train models on.
use_E_cstr = False  # Adds energy constraints to the kernel. `False` usually provides better performance. `True` can help converge training in rare cases.
overwrite = True  # Whether the script will overwrite a model.
torch = False  # Whether to use torch to accelerate training.
max_processes = None
model_name = f'140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-randomtrain{num_train}'

# None for automatically selected or specify an array.
idxs_train = None



###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if idxs_train is not None:
    assert num_train == len(idxs_train)

def main():
    os.chdir(save_dir)
    
    train = mbGDMLTrain()
    train.load_dataset(dataset_path)
    train.train(
        model_name, num_train, num_validate, num_test, solver='analytic',
        sigmas=sigmas, save_dir='.', use_sym=True, use_E=True,
        use_E_cstr=use_E_cstr, use_cprsn=False, idxs_train=idxs_train,
        max_processes=max_processes, overwrite=overwrite, torch=torch,
    )

main()