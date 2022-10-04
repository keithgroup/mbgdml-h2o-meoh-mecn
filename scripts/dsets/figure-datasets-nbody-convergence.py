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
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mbgdml.data import dataSet
from mbgdml.criteria import cm_distance_sum
from reptar import File

plot_data = {
    '16h2o-2body-wrt-l' : {
        'exdir_path': '16h2o-yoo.etal.exdir',
        'group_key': 'samples_2h2o',
        'Z_key': 'atomic_numbers',
        'R_key': 'geometry',
        'entity_key': 'entity_ids',
        'energy_key': 'energy_ele_nbody_mp2.def2tzvp_orca',
        'color': '#4ABBF3',
        'mb_order': 2,
    },
    '16h2o-3body-wrt-l' : {
        'exdir_path': '16h2o-yoo.etal.exdir',
        'group_key': 'samples_3h2o',
        'Z_key': 'atomic_numbers',
        'R_key': 'geometry',
        'entity_key': 'entity_ids',
        'energy_key': 'energy_ele_nbody_mp2.def2tzvp_orca',
        'color': '#4ABBF3',
        'mb_order': 3,
    },
    '16mecn-2body-wrt-l' : {
        'exdir_path': '16mecn-remya.etal.exdir',
        'group_key': 'samples_2mecn',
        'Z_key': 'atomic_numbers',
        'R_key': 'geometry',
        'entity_key': 'entity_ids',
        'energy_key': 'energy_ele_nbody_mp2.def2tzvp_orca',
        'color': '#61BFA3',
        'mb_order': 2,
    },
    '16mecn-3body-wrt-l' : {
        'exdir_path': '16mecn-remya.etal.exdir',
        'group_key': 'samples_3mecn',
        'Z_key': 'atomic_numbers',
        'R_key': 'geometry',
        'entity_key': 'entity_ids',
        'energy_key': 'energy_ele_nbody_mp2.def2tzvp_orca',
        'color': '#61BFA3',
        'mb_order': 3,
    },
    '16meoh-2body-wrt-l' : {
        'exdir_path': '16meoh-pires.deturi.exdir',
        'group_key': 'samples_2meoh',
        'Z_key': 'atomic_numbers',
        'R_key': 'geometry',
        'entity_key': 'entity_ids',
        'energy_key': 'energy_ele_nbody_mp2.def2tzvp_orca',
        'color': '#FFB5BA',
        'mb_order': 2,
    },
    '16meoh-3body-wrt-l' : {
        'exdir_path': '16meoh-pires.deturi.exdir',
        'group_key': 'samples_3meoh',
        'Z_key': 'atomic_numbers',
        'R_key': 'geometry',
        'entity_key': 'entity_ids',
        'energy_key': 'energy_ele_nbody_mp2.def2tzvp_orca',
        'color': '#FFB5BA',
        'mb_order': 3,
    },
}
save_dir = 'analysis/nbody-energy-wrt-l'

solvent_colors = {'h2o': '#4ABBF3', 'mecn': '#61BFA3', 'meoh': '#FFB5BA'}
cluster_num_color = '#7D7D7D'

cutoff_grid_step = 0.01
desired_percent_error = 0.01  # None or float (e.g., 0.01)
desired_energy_error = None  # None or float

# More information: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
use_rc_params = True
font_dirs = ['../../fonts/roboto']
rc_json_path = '../matplotlib-rc-params.json'




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
data_dir = os.path.join(base_dir, 'data/isomers/')
save_dir = os.path.join(base_dir, save_dir)
os.makedirs(save_dir, exist_ok=True)

hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant

# Setup matplotlib style
if use_rc_params:
    import json
    with open(rc_json_path, 'r') as f:
        rc_params = json.load(f)
    font_paths = mpl.font_manager.findSystemFonts(
        fontpaths=font_dirs, fontext='ttf'
    )
    for font_path in font_paths:
        mpl.font_manager.fontManager.addfont(font_path)
    for key, params in rc_params.items():
        plt.rc(key, **params)


for plot_name, data in plot_data.items():
    print(f'Working on {plot_name}')
    exdir_path = os.path.join(data_dir, data['exdir_path'])
    rfile = File(exdir_path, 'r')

    group_key = data['group_key']
    Z = rfile.get(f'{group_key}/{data["Z_key"]}')
    R = rfile.get(f'{group_key}/{data["R_key"]}')
    E = rfile.get(f'{group_key}/{data["energy_key"]}')
    entity_ids = rfile.get(f'{group_key}/{data["entity_key"]}')
    color = data['color']
    mb_order = data['mb_order']

    E *= hartree2kcalmol

    # Calculate size metric
    L = np.empty(len(R))
    for i in range(len(R)):
        _, l = cm_distance_sum(Z, R[i], None, entity_ids)
        L[i] = l


    # Generates range of L cutoffs.
    L_lower_bound = math.floor(np.min(L) * 10)/10.0  # Rounded down to tenths
    L_upper_bound = math.ceil(np.max(L) * 10)/10.0  # Rounded up to tenths
    cutoff_range = np.arange(
        L_lower_bound, L_upper_bound + cutoff_grid_step, cutoff_grid_step
    )

    # Computes n-body energy at each cutoff and the number of clusters.
    nbody_energy_wrt_L = np.empty(cutoff_range.shape)
    num_clusters_wrt_L = np.empty(cutoff_range.shape)
    for i in range(len(cutoff_range)):
        R_idx_include = np.where(L < cutoff_range[i])[0]
        nbody_energy_wrt_L[i] = np.sum(E[R_idx_include])
        num_clusters_wrt_L[i] = int(len(R_idx_include))
    nbody_energy_total = np.sum(E)
    print(f'Total n-body energy: {nbody_energy_total:.1f} kcal/mol')
    nbody_energy_error_wrt_L = nbody_energy_wrt_L - nbody_energy_total


    # Computes smallest cutoff that provides desired error.
    if desired_percent_error is not None:
        max_abs_energy_error = abs(desired_percent_error * nbody_energy_total)
    elif desired_energy_error is not None:
        max_abs_energy_error = desired_energy_error

    idxs_lower_error = np.argwhere(
        np.abs(nbody_energy_error_wrt_L) < max_abs_energy_error
    ).T[0]
    L_cutoff = None
    for i in reversed(range(len(idxs_lower_error)-1)):
        idx_i = idxs_lower_error[i]
        idx_j = idxs_lower_error[i+1]
        if idx_j - idx_i > 1:
            L_cutoff = idx_j
            break
    if L_cutoff is None:
        L_cutoff = idxs_lower_error[0]
    
    print(f'Min. cutoff: {cutoff_range[L_cutoff]:.1f}')
    print(f'Energy error at cutoff: {nbody_energy_error_wrt_L[L_cutoff]:.3f}')

    ###   FIGURE   ###

    fig, ax = plt.subplots(1, 1 , figsize=(3.5, 2.6), constrained_layout=True)

    # N-body energy
    ax.plot(
        cutoff_range, nbody_energy_wrt_L,
        color=color,
        markersize=0, 
        linestyle='-', linewidth=1.5,
        label='$n$-body energy',
        zorder=1
    )

    # Number of clusters
    ax2 = ax.twinx()
    ax2.plot(
        cutoff_range, num_clusters_wrt_L,
        color=cluster_num_color,
        marker='o', markersize=0,
        linestyle='-', linewidth=1.5,
        label='# clusters'
    )

    # Axis labels
    ax.set_xlabel('L cutoff (Ang.)')
    ax.set_ylabel(f'{mb_order}-body energy (kcal/mol)', color=color)
    ax2.set_ylabel(f'Number of clusters', color=cluster_num_color)

    # Axis tick label colors
    ax.tick_params(axis='y', colors=color)
    ax2.tick_params(axis='y', colors=cluster_num_color)

    # Reorders axis
    ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)

    plt_path = os.path.join(save_dir, plot_name+'.png')
    plt.savefig(plt_path, dpi=1000)
    plt.close()
    print()


