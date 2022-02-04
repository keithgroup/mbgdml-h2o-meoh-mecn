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
Removes data set n-body contributions using lower-order model predictions.
"""

import os
import numpy as np
from mbgdml.data import dataSet

##### Job Information #####

# Model paths.
model_dir = '../../data/models'
model_dir_h2o = f'{model_dir}/h2o'

model_dir_solvent = model_dir_h2o
model_paths = [
    f'{model_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.1h2o-model-train500.npz',
    f'{model_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.2h2o-model.mb-train500.npz'
]

# Data set paths.
dataset_dir = '../../data/datasets'
dataset_dir_h2o = f'{dataset_dir}/h2o'

dataset_path_original = f'{dataset_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o-dset.npz'

# Training information.
num_train = 500

# Script options.
overwrite = True





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def get_filename(path):
    """The name of the file without the extension from a path.

    If there are periods in the file name with no file extension, will always
    remove the last one.

    Parameters
    ----------
    path : :obj:`str`
        Path to file.

    Returns
    -------
    :obj:`str`
        The file name without an extension.
    """
    return os.path.splitext(os.path.basename(path))[0]

def create_mb_dataset(dataset_name, dataset_path, model_paths):
    """

    Parameters
    ----------
    dataset_path : :obj:`str`
        Path to data set.
    model_paths : :obj:`list` [:obj:`str`]
        Paths to saved many-body GDML models in the form of
        :obj:`numpy.NpzFile`.
    """
    ref_dataset = dataSet(dataset_path)
    mb_dataset = dataSet()
    mb_dataset.create_mb_from_models(ref_dataset, model_paths)
    mb_dataset.name = dataset_name
    return mb_dataset

def main():
    dataset_mb_name = get_filename(dataset_path_original) + f'.mb-train{num_train}'
    save_dir = os.path.dirname(dataset_path_original) + '/'
    
    if not os.path.exists(save_dir + dataset_mb_name + '.npz') or overwrite:
        print(f'Making {dataset_mb_name} ...')
        mb_dataset = create_mb_dataset(
            dataset_mb_name, dataset_path_original, model_paths
        )
        mb_dataset.save(mb_dataset.name, mb_dataset.dataset, save_dir)
    else:
        print(f'{dataset_mb_name} already exists, and overwrite is False.')
    
    

main()