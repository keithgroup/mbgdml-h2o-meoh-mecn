#!/usr/bin/env python3

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

"""Write combined trajectory as a npy file."""

import os
import numpy as np
from numpy.lib.format import open_memmap
from mbgdml.utils import get_entity_ids, get_comp_ids
from reptar import File

exdir_path = 'h2o/137h2o-mbgdml-md.exdir'
group_keys = ['1-nvt', '2-nvt']
npy_path = 'h2o/137h2o-mbgdml-nvt_1_2.npy'

trim_restart_initial = True





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../data/md'
exdir_path = os.path.join(base_dir, exdir_path)
npy_path = os.path.join(base_dir, npy_path)

print('Loading data')
rfile = File(exdir_path)

R = []
for i in range(len(group_keys)):
    group = rfile.get(group_keys[i])

    Z = group.get('atomic_numbers')
    R.append(group.get('geometry'))


# For restarted MD simulations the last structure is also the initial
# structure of the restarted MD simulation. We need to remove this structure
# from the restarts.
if trim_restart_initial and len(group_keys) > 1:
    for i in range(1, len(group_keys)):
        R[i] = R[i][1:]

print('Writing file')
R = np.array(R, dtype=object)
R = np.vstack(R).astype(np.double)
np.save(npy_path, R)


