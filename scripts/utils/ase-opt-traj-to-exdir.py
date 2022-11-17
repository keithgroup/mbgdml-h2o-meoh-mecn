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

"""Creates an exdir group from an ASE trajectory."""

from mbgdml.utils import get_entity_ids, get_comp_ids
import os
from reptar import Creator
import qcelemental as qcel


traj_path = 'mecn/122mecn/0-opt/122mecn-mbgdml-opt.traj'
group_key = '0-opt'  # Key to add traj under

n_molecules = 122
comp_id = 'mecn'
entity_ids = get_entity_ids(6, n_molecules)
comp_ids = get_comp_ids(comp_id, n_molecules, entity_ids)

exdir_path = 'md/mecn/122mecn-mbgdml-md.exdir'

# Keys and values to add to new group
extra_adds = (
    ('optimizer_algo', 'BGFS'),
    ('optimizer_force_max', 0.2 * qcel.constants.conversion_factor('eV', 'hartree')),
    ('periodic_mic_cutoff', 11.0),
    ('readme',
    'Preliminary energy minimization before MD simulation. '
    'Periodic box using mbGDML ASE calculator. '
    'The optimization was terminated after 204 steps due to little progress.')
)

###   SCRIPT   ###
# Handles paths
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
traj_dir = '../../../mbgdml-h2o-meoh-mecn-md'
data_dir = os.path.join(base_dir, 'data/')
traj_path = os.path.join(traj_dir, traj_path)
exdir_path = os.path.join(data_dir, exdir_path)


create = Creator()
create.load(exdir_path, mode='w', allow_remove=True)
rfile = create.from_calc(group_key, traj_path=traj_path)

for stuff in extra_adds:
    key, data = stuff
    rfile.put(f'{group_key}/{key}', data)
