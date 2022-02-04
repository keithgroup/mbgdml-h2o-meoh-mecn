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

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
from mbgdml.data import dataSet
from mbgdml import criteria


# Data set information
dset_dir = '../../data/datasets'
dset_dir_h2o = f'{dset_dir}/h2o'
dset_dir_mecn = f'{dset_dir}/mecn'
dset_dir_meoh = f'{dset_dir}/meoh'

dset_dir_solvent = dset_dir_meoh
dset_path = f'{dset_dir_solvent}/4meoh/16meoh.pires.deturi.4meoh-dset.mb.npz'
# 16h2o.yoo.etal.boat.b.4h2o-dset.mb
# 16meoh.pires.deturi.2meoh-dset.mb

save_dir = '../../analysis/mb-convergence'

solvent = 'meoh'

# Plot
plot_name = '16mecn.remya.etal.2mecn-dset.mb-cm.distance.sum.convergence'

error_line_x = 9
error_line_y = 30

solvent_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}
cluster_num_color = 'silver'

save = False

criteria_selection = criteria.cm_distance_sum


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if save_dir[-1] != '/':
    save_dir += '/'
save_dir += solvent

# z_slice atoms for each possible species (starting from zero).
criteria_molecule_index = {
    'h2o': 0,
    'mecn': 4,
    'meoh': 0
}





dset = dataSet(dset_path)

z_slice = []
criteria_name = criteria_selection.__name__
if criteria_name == 'distance_sum' or criteria_name == 'distance_all':
    z_slice = criteria.get_z_slice(
        dset.entity_ids, dset.comp_ids, criteria_molecule_index
    )

# Calculate distance_sum for each structure.
criteria = np.empty((dset.R.shape[0]))
for i in range(dset.R.shape[0]):
    _, distance = criteria_selection(dset.z, dset.R[i], z_slice, dset.entity_ids)
    criteria[i] = distance

# Generates range of criteria cutoffs.
criteria_low_bound = math.floor(np.min(criteria) * 10)/10.0  # Rounded down to tenths
criteria_high_bound = math.ceil(np.max(criteria) * 10)/10.0  # Rounded up to tenths

step = 0.01
cutoff_range = np.arange(criteria_low_bound, criteria_high_bound + step, step)

# Computes n-body energy at each cutoff and the number of clusters.
nbody_energy = np.empty(cutoff_range.shape)
num_clusters = np.empty(cutoff_range.shape)
for i in range(len(cutoff_range)):
    R_cutoff_idx = np.where(criteria < cutoff_range[i])[0]
    nbody_energy[i] = np.sum(dset.E[R_cutoff_idx])
    num_clusters[i] = int(len(R_cutoff_idx))
nbody_energy_ref = np.sum(dset.E)
print(f'Total n-body energy: {nbody_energy_ref:.1f} kcal/mol')

# Computes largest cutoff that provides 1 % error.
max_percent_error = 1
max_error_abs = abs((max_percent_error/100) * nbody_energy_ref)
for i in reversed(range(len(cutoff_range))):
    error_abs = abs(nbody_energy_ref - nbody_energy[i])
    if error_abs > max_error_abs:
        error_index = i + 1
        break
max_error_cutoff = cutoff_range[error_index]

###   FIGURE   ###
# Setting up general figure properties
font = {'family' : 'sans-serif',
        'size'   : 8}
plt.rc('font', **font)

fig, ax = plt.subplots(1, 1 , figsize=(3.5, 2.6), constrained_layout=True)

# N-body energy
ax.plot(
    cutoff_range, nbody_energy,
    color=solvent_colors[solvent],
    marker='o', markersize=0, 
    linestyle='-', linewidth=1.5,
    label='$n$-body energy',
    zorder=1
)

# Number of clusters
ax2 = ax.twinx()
ax2.plot(
    cutoff_range, num_clusters,
    color=cluster_num_color,
    marker='o', markersize=0,
    linestyle='-', linewidth=1.5,
    label='# clusters'
)

# Max error line
ax.axvline(
    max_error_cutoff,
    color='lightgrey',
    linestyle='--', linewidth=1,
    zorder=0
)
ax2.text(
    error_line_x, error_line_y, f'{max_percent_error}% Error\n$L$ = {max_error_cutoff:.2f}',
    color='lightgrey'
)

# Axis labels
ax.set_xlabel('$L$ cutoff ($\AA$)')
ax.set_ylabel(f'{dset.mb}-body energy (kcal/mol)', color=solvent_colors[solvent])
ax2.set_ylabel(f'Number of clusters', color=cluster_num_color)

# Axis tick label colors
ax.tick_params(axis='y', colors=solvent_colors[solvent])
ax2.tick_params(axis='y', colors=cluster_num_color)

# Reorders axis
ax.set_zorder(ax2.get_zorder()+1)
ax.patch.set_visible(False)

plt_path = f'{save_dir}/{plot_name}.png'
if save:
    print(f'Saving {plt_path}')
    plt.savefig(plt_path, dpi=1000)


