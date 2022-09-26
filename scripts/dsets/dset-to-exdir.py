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

"""Add data set information to exdir file"""

import os
import numpy as np
from reptar import File
from reptar.utils import gen_entity_ids, gen_comp_ids, get_md5
from reptar import __version__ as reptar_version

# Exdir
dir_path_exdir = '/home/alex/Dropbox/keith/projects/mbgdml-h2o-meoh-mecn/data'
exdir_path = '140h2o-xtb.md-samples.exdir'

# Data set
dir_path_npz = '/home/alex/repos/keith/mbgdml-h2o-meoh-mecn/data/datasets'
dset_path = 'h2o/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o-dset.npz'

group_name = '2h2o'
exdir_mode = 'a'
allow_remove = False

entity_ids = gen_entity_ids(3, 2)
comp_ids = gen_comp_ids('h2o', 2, entity_ids)
print(entity_ids)
print(comp_ids)

use_2 = False  # Append data
dset_path_2 = 'h2o/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o-dset.npz'

# Hartree conversion: https://github.com/cclib/cclib/blob/6ba33bdd48bf336a81400147d2870323b241e457/cclib/parser/utils.py#L107


###   Script   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

dset_path = os.path.join(dir_path_npz, dset_path)
exdir_path = os.path.join(dir_path_exdir, exdir_path)
npz_data = File(dset_path, mode='r')
npz_dict = npz_data.as_dict('/')

if use_2:
    dset_path_2 = os.path.join(dir_path_npz, dset_path_2)
    npz_data_2 = File(dset_path_2, mode='r')
    npz_dict_2 = npz_data_2.as_dict('/')

# Get arrays
R = npz_dict['R']
E = npz_dict['E']
E /= 627.50947414  # Hartree
F = npz_dict['F']
F /= 627.50947414  # Hartree/A
G = -F
r_prov_specs = npz_dict['r_prov_specs']

if use_2:
    R_2 = npz_dict_2['R']
    E_2 = npz_dict_2['E']
    E_2 /= 627.50947414  # Hartree
    F_2 = npz_dict_2['F']
    F_2 /= 627.50947414  # Hartree/A
    G_2 = -F_2
    r_prov_specs_2 = npz_dict_2['r_prov_specs']


    # Combining data
    R_all = R_2.tolist()
    E_all = E_2.tolist()
    G_all = G_2.tolist()
    r_prov_specs_all = r_prov_specs_2.tolist()


    for i in range(len(R)):
        test_r_prov_specs = r_prov_specs[i]
        if (np.array(r_prov_specs_all)==test_r_prov_specs).all(1).any():
            continue
        R_all.append(R[i])
        E_all.append(E[i])
        G_all.append(G[i])
        r_prov_specs_all.append(test_r_prov_specs)

    R_all = np.array(R_all)
    E_all = np.array(E_all)
    G_all = np.array(G_all)
    r_prov_specs_all = np.array(r_prov_specs_all)
else:
    R_all = R
    E_all = E
    G_all = G
    r_prov_specs_all = r_prov_specs


exdir_data = File(exdir_path, mode=exdir_mode, allow_remove=allow_remove)
exdir_data.init_group(group_name)

exdir_data.add(f'{group_name}/r_prov_ids', r_prov_specs_all)
exdir_data.add(f'{group_name}/r_prov_specs', r_prov_specs_all)
exdir_data.add(f'{group_name}/atomic_numbers', npz_dict['z'])
exdir_data.add(f'{group_name}/geometry', R_all)
exdir_data.add(f'{group_name}/entity_ids', entity_ids)
exdir_data.add(f'{group_name}/comp_ids', comp_ids)
exdir_data.add(f'{group_name}/energy_ele_mp2.def2tzvp_orca', E_all)
exdir_data.add(f'{group_name}/grads_mp2.def2tzvp_orca', G_all)


comp_ids_num = {}
unique_comp_ids, comp_ids_freq = np.unique(comp_ids, return_counts=True)
for comp_id, num in zip(unique_comp_ids, comp_ids_freq):
    comp_ids_num[str(comp_id)] = int(num)
exdir_data.add(f'{group_name}/comp_ids_num', comp_ids_num)


# MD5 stuff
md5 = get_md5(exdir_data, group_name)
exdir_data.add(f'{group_name}/md5', md5)

try:
    md5_arrays = get_md5(exdir_data, group_name, only_arrays=True)
    exdir_data.add(f'{group_name}/md5_arrays', md5_arrays)
except Exception:
    pass

try:
    md5_structures = get_md5(exdir_data, group_key, only_structures=True)
    exdir_data.add(f'{group_name}/md5_structures', md5_structures)
except Exception:
    pass

exdir_data.add(f'{group_name}/reptar_version', reptar_version)
