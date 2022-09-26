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

"""Adds many-body energies and forces from sGDML dataset format to exdir."""

import os
import numpy as np
from reptar import File
from reptar.utils import gen_entity_ids, gen_comp_ids, get_md5
from reptar import __version__ as reptar_version

# Exdir
dir_path_exdir = '../../data'
exdir_path = '140h2o-xtb.md-samples.exdir'

# Many-body dataset
dir_path_npz = '../../data/datasets'
dset_path = 'h2o/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o-dset.mb-cm10.npz'

group_key = '3h2o'
exdir_mode = 'a'
allow_remove = False

print(group_key)

###   Script   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

dset_path = os.path.join(dir_path_npz, dset_path)
exdir_path = os.path.join(dir_path_exdir, exdir_path)
npz_file = File(dset_path, mode='r')
npz_dict = npz_file.as_dict('/')

# Get arrays
E_mb = npz_dict['E']
E_mb /= 627.50947414  # Hartree
F_mb = npz_dict['F']
F_mb /= 627.50947414  # Hartree/A
G_mb = -F_mb
r_prov_specs = npz_dict['r_prov_specs']

rfile = File(exdir_path, mode=exdir_mode, allow_remove=allow_remove)

assert np.array_equal(r_prov_specs, rfile.get(f'{group_key}/r_prov_specs'))

rfile.add(f'{group_key}/energy_ele_nbody_mp2.def2tzvp_orca', E_mb)
rfile.add(f'{group_key}/grads_nbody_mp2.def2tzvp_orca', G_mb)
