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


traj_path = 'meoh/61meoh/3-nvt/61meoh-mbgdml-nvt-beren-init_100-taut_0.1-3.traj'
group_key = '3-nvt'  # Key to add traj under

n_molecules = 61
comp_id = 'meoh'
entity_ids = get_entity_ids(6, n_molecules)
comp_ids = get_comp_ids(comp_id, n_molecules, entity_ids)

exdir_path = 'md/meoh/61meoh-mbgdml-md.exdir'

# Keys and values to add to new group
extra_adds = (
    # ('velcs_init_algo', 'Maxwell Boltzmann'),
    # ('velcs_init_temp', 100),
    ('thermostat_type', 'Berendsen'),
    ('thermostat_temp', 298.15),
    ('thermostat_beren_tau_temp', 0.1),
    ('t_step', 1.0),
    ('md_duration', 10.0),
    ('md_steps_dump_traj', 1),
    ('periodic_mic_cutoff', 8.0),
    ('readme',
    'A 10 ps NVT simulation at 298.15 K using the Berendsen thermostat. '
    'Initialized from 2-nvt trajectory. '
    )
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
create.load(exdir_path, mode='a', allow_remove=False)
rfile = create.from_calc(group_key, traj_path=traj_path)
rfile.put(f'{group_key}/entity_ids', entity_ids)
rfile.put(f'{group_key}/comp_ids', comp_ids)

for stuff in extra_adds:
    key, data = stuff
    rfile.put(f'{group_key}/{key}', data)
