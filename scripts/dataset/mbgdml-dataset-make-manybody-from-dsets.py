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
Removes data set n-body contributions using lower-order data sets.
"""

from mbgdml.data import dataSet
import os
import numpy as np
from mbgdml.data import dataSet

##### Job Information #####

# Dset dir
dset_dir = '../../data/datasets'
dset_dir_h2o = f'{dset_dir}/h2o'
dset_dir_mecn = f'{dset_dir}/mecn'
dset_dir_meoh = f'{dset_dir}/meoh'

dset_dir_solvent = dset_dir_h2o

# Reference dset path.
ref_dset_path = f'{dset_dir_solvent}/3h2o/temelso.etal/4h2o.temelso.etal.dset.3h2o-dset-mp2.augccpvtz.npz'

# Lower order many-body data sets.
lower_dset_paths = [
    f'{dset_dir_solvent}/1h2o/temelso.etal/4h2o.temelso.etal.dset.1h2o-dset-mp2.augccpvtz.npz',
    f'{dset_dir_solvent}/2h2o/temelso.etal/4h2o.temelso.etal.dset.2h2o-dset.mb-mp2.augccpvtz.npz',
]

dataset_mb_name = '4h2o.temelso.etal.dset.3h2o-dset.mb-mp2.augccpvtz'

# Script options.
overwrite = False



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

def create_mb_dataset(dset_name, ref_dset_path, lower_dset_paths):
    """

    Parameters
    ----------
    dset_name : :obj:`str`

    dataset_path : :obj:`str`
        Path to data set.
    model_paths : :obj:`list` [:obj:`str`]
        Paths to saved many-body GDML models in the form of
        :obj:`numpy.NpzFile`.
    """
    ref_dset = dataSet(ref_dset_path)
    mb_dataset = dataSet()
    mb_dataset.create_mb_from_dsets(ref_dset, lower_dset_paths)
    mb_dataset.name = dset_name
    return mb_dataset

def main():
    save_dir = os.path.dirname(ref_dset_path) + '/'
    
    if not os.path.exists(save_dir + dataset_mb_name + '.npz') or overwrite:
        print(f'Making {dataset_mb_name} ...')
        mb_dataset = create_mb_dataset(
            dataset_mb_name, ref_dset_path, lower_dset_paths
        )
        mb_dataset.save(mb_dataset.name, mb_dataset.asdict, save_dir)
    else:
        print(f'{dataset_mb_name} already exists, and overwrite is False.')
    

main()