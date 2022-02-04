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
Used to quickly develop code for other scripts.
"""

import os
import numpy as np
from mbgdml.data import dataSet
from mbgdml.predict import mbPredict
from mbgdml.utils import get_filename

data_dir = '../../data'
md_path = f'{data_dir}/md/meoh/6meoh/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k.npz'
dset_path = f'{data_dir}/datasets/meoh/6meoh/6meoh.boyd.etal.1.md.orca.mp2.def2tzvp.300k-dset.npz'



###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

dset = dataSet(dset_path)
dset_name = get_filename(dset_path)
dset_dir = os.path.dirname(dset_path)
md = dict(np.load(md_path, allow_pickle=True))



dset.E = md['E_potential']
dset.F = md['F']
dset.e_unit = 'kcal/mol'
dset.theory = 'mp2.def2tzvp.frozencore'
dset.name = dset_name

dset.save(dset.name, dset.asdict, dset_dir)
