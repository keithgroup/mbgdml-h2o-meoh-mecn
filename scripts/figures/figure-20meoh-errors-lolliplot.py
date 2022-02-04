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

import numpy as np
import matplotlib.pyplot as plt
from mbgdml.data import predictSet
from cclib.parser.utils import convertor

solvent = 'meoh'
max_nbody_order = 3

# Predict set paths.
pset_dir = '../../data/predictsets'
pset_dir_h2o = f'{pset_dir}/h2o'
pset_dir_mecn = f'{pset_dir}/mecn'
pset_dir_meoh = f'{pset_dir}/meoh'

pset_20meoh = f'{pset_dir_meoh}/20meoh/20meoh.yao.etal.all-pset-62meoh.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'

save_dir = '../../analysis/20meoh'
plot_name = '20meoh-yao2017many-comparison'

if save_dir[-1] != '/':
    save_dir += '/'

solvent_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}
ref_color = 'silver'


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))


# Manuscript (DOI: 10.1063/1.4973380) data from SI.

yao2017many_20meoh_e_mp2 = np.array(  # Eh
    [-2311.00235, -2310.91947, -2310.93719, -2310.95301, -2310.92133]
)
yao2017many_20meoh_e_mp2_mbe = np.array(  # Eh
    [-2310.93744, -2310.86004, -2310.87428, -2310.89366, -2310.86089]
)
yao2017many_20meoh_e_mbe_nn = np.array(  # Eh
    [-2310.93545, -2310.85729, -2310.87199, -2310.89384, -2310.85704]
)
yao2017many_20meoh_e_mp2 = convertor(
    yao2017many_20meoh_e_mp2, 'hartree', 'kcal/mol'
)
yao2017many_20meoh_e_mp2_mbe = convertor(
    yao2017many_20meoh_e_mp2_mbe, 'hartree', 'kcal/mol'
)
yao2017many_20meoh_e_mbe_nn = convertor(
    yao2017many_20meoh_e_mbe_nn, 'hartree', 'kcal/mol'
)

# mbGDML data
pset = predictSet(pset_20meoh)
e_mbgdml_20meoh = pset.nbody_predictions([1, 2, 3])[0]
# All energies are in kcal/mol at this point.

# Getting relative energies.
relative_energies = {}
method_keys = ['RI-MP2', 'MBE', 'NN-MBE', 'mbGDML']

cluster_sort_idx = np.argsort(yao2017many_20meoh_e_mp2)
for key,energies in zip(
    method_keys,
    (yao2017many_20meoh_e_mp2, yao2017many_20meoh_e_mp2_mbe,
     yao2017many_20meoh_e_mbe_nn, e_mbgdml_20meoh)
):
    ref_energy = energies[cluster_sort_idx[0]]
    sorted_energies = energies[cluster_sort_idx[1:]]
    relative_energies[key] = sorted_energies - ref_energy

relative_energies_errors = {}
relative_energies_ref = relative_energies['RI-MP2']
for key,energies in relative_energies.items():
    if key != 'RI-MP2':
        relative_energies_errors[key] = energies - relative_energies_ref

##################
###   FIGURE   ###
##################

# Setting up general figure properties
font = {'family' : 'sans-serif',
        'size'   : 8}
plt.rc('font', **font)

fig, ax = plt.subplots(1, 1, figsize=(3.25, 3.25), constrained_layout=True)

method_labels = ['MBE', 'NN-MBE', 'mbGDML']
method_markers = ['o', 's', 'v']
# sys_color = solvent_colors[solvent]
sys_color = 'darkgrey'
marker_size = 5
line_width = 1.5



method_offset = 0.1
structure_offset = 0.4


for i in range(len(method_keys[1:])):
    method_key = method_keys[i+1]
    method_label = method_labels[i]
    rel_e_error = np.abs(relative_energies_errors[method_key])
    
    x_0 = i*method_offset
    x_values = [x_0 + structure_offset*j for j in range(len(rel_e_error))]

    for j in range(len(rel_e_error)):
        if j == 0:
            kwargs = {'label': method_label}
        else:
            kwargs = {}
        ax.plot(  # Vertical line.
            [x_values[j], x_values[j]], [0, rel_e_error[j]],
            marker='', markersize=0, 
            linestyle='-', linewidth=line_width,
            color=sys_color
        )
        ax.plot(
            x_values[j], rel_e_error[j],
            marker=method_markers[i], markersize=marker_size, 
            linestyle='', linewidth=0,
            color=sys_color, markeredgecolor=None,
            zorder=1, **kwargs
        )

ax.set_ylabel('Relative Energy Error (kcal/mol)')
ax.set_ylim(bottom=0)

ax.set_xlabel('20meoh Isomers')
num_structures = len(relative_energies_errors['MBE'])
xtick_midpoint_position = (len(method_labels)*method_offset)/2-method_offset/2
print(xtick_midpoint_position)
xticks = [xtick_midpoint_position + structure_offset*j for j in range(num_structures)]
ax.set_xticks(xticks)
ax.set_xticklabels([f'Isomer {i}' for i in range(1, num_structures+1)])

ax.legend(loc='best', frameon=False)  
plt_path = f'{save_dir}{plot_name}.png'
print(f'Saving {plt_path}')
plt.savefig(plt_path, dpi=1000)


# Printing information

print('Relative energies in kcal/mol')
for key,value in relative_energies.items():
    print(key, ': ', value)
