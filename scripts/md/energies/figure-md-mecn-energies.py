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

"""Plots the RMSDs of MD simulations"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter


fig_save_type = 'svg'  # eps or png
figsize = (6, 6)
font_size = 18  # 8
line_width = 1.5

save = True
plot_name = '6mecn-ase.md-energies'
save_dir = 'analysis/md/6mer/energies/'

idxs = [0, 1, 2, 3, 4, 5, 6]  # 0 - 6

md_npz_paths = [
    [
        'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-0/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-0.npz',
        'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-1/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-1.npz',
        'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-2/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-2.npz',
        'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-3/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-3.npz',
    ],
    [
        'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.rimp2.def2tzvp.300k-0/6mecn-ase.md-orca.rimp2.def2tzvp.300k-0.npz',
        'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.rimp2.def2tzvp.300k-0-1/6mecn-ase.md-orca.rimp2.def2tzvp.300k-0-1.npz',
    ],
    'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2svp.300k-0-0/6mecn-ase.md-orca.mp2.def2svp.300k-0-0.npz',
    'data/md/mecn/6mecn/gdml/6mecn-ase.md-gdml.train1000.300k-0/6mecn-ase.md-mbgdml.train1000.300k-0.npz',
    'data/md/mecn/6mecn/gap/6mecn-ase.md-gap.train1000.300k-0/6mecn-ase.md-gap.train1000.300k-0.npz',
    'data/md/mecn/6mecn/schnet/6mecn-ase.md-schnet.train1000.300k-0/6mecn-ase.md-schnet.train1000.300k-0.npz',
    'data/md/mecn/6mecn/gfn2/6mecn-ase.md-gfn2.300k-0/6mecn-ase.md-gfn2.300k-0.npz',
]
labels = [
    'MP2',
    'RI-MP2',
    'def2-SVP',
    'mbGDML',
    'mbGAP',
    'mbSchNet',
    'GFN2-xTB'
]
colors = [
    '#343a40',
    '#6c757d',
    '#ced4da',
    '#1b4332',
    '#2d6a4f',
    '#52b788',
    '#95d5b2',
    '#d8f3dc',
]
r_alpha = 1.0
m_alpha = 1.0
alphas = [
    r_alpha,
    m_alpha,
    m_alpha,
    m_alpha,
    m_alpha,
    m_alpha,
    m_alpha,
]

# More information: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
use_rc_params = True
font_dirs = ['../../../fonts/roboto']
rc_json_path = '../../matplotlib-rc-params.json'





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
save_dir = os.path.join(base_dir, save_dir)
os.makedirs(save_dir, exist_ok=True)

# Collects (and possibly combines) all Cartesian coordinates.
md_data = []
for i in idxs:
    md_paths = md_npz_paths[i]
    
    # Single path, no splits in MD trajectory.
    if isinstance(md_paths, str):
       md_paths = [os.path.join(base_dir, md_paths)]
    elif isinstance(md_paths, list):
        md_paths = [os.path.join(base_dir, j) for j in md_paths]

    md_E = [
        dict(np.load(md_path, allow_pickle=True))['E_potential'] for md_path in md_paths
    ]
    # For restarted MD simulations the last structure is also the initial
    # structure of the restarted MD simulation. We need to remove this structure
    # from the restarts.
    if len(md_E) > 1:
        for j in range(1, len(md_E)):
            md_E[j] = md_E[j][1:]

    md_E = np.array(md_E, dtype=object)
    md_E = np.hstack(md_E).astype(np.double)
    md_data.append(md_E)


# Compute relative energies

E_rel_data = []
for E in md_data:
    E_rel = np.zeros(E.shape, dtype=np.double)
    for i in range(len(E_rel)):
        E_rel[i] = E[i] - E[0]
    
    # Smooth data
    window_length = 101  # must be odd
    E_rel = savgol_filter(E_rel, window_length, 4)

    E_rel_data.append(E_rel)

E_maes, E_rmses, E_sses = [], [], []
for E_rel in E_rel_data:
    E_errors = E_rel - E_rel_data[0]
    
    E_mae = np.mean(np.abs(E_errors))
    E_maes.append(E_mae)

    E_rmse = np.sqrt(np.mean((E_errors)**2))
    E_rmses.append(E_rmse)

    E_sse = np.dot(E_errors, E_errors)
    E_sses.append(E_sse)
        


# Plotting figure

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


fig, ax = plt.subplots(1, 1 , figsize=figsize, constrained_layout=True)

for i in range(len(idxs)):
    E_rel = E_rel_data[i]
    label = labels[idxs[i]]
    color = colors[idxs[i]]
    alpha = alphas[idxs[i]]

    if label == '':
        label = None
    ax.plot(
        E_rel,
        color=color,
        marker='', markersize=0, 
        linestyle='-', linewidth=line_width,
        label=label,
        alpha=alpha,
    )

    print(f'{label} MAE: {E_maes[i]:.5f}')
    print(f'{label} RMSE: {E_rmses[i]:.5f}')
    print(f'{label} SSE: {E_sses [i]:.5f}\n')

# X axis
ax.set_xlabel('Time (fs)')
ax.set_xlim(xmin=0)

# Y axis
ax.set_ylabel('$\Delta$ E (kcal/mol)')

ax.legend(frameon=False)

if save:
    plt_path = os.path.join(save_dir, plot_name)

    if fig_save_type == 'png':
        plt_path += '.png'
        print(f'Saving {plt_path}')
        plt.savefig(plt_path, format=fig_save_type, dpi=1200)
    elif fig_save_type == 'eps':
        plt_path += '.eps'
        print(f'Saving {plt_path}')
        plt.savefig(plt_path, format=fig_save_type, dpi=1200)
    elif fig_save_type == 'svg':
        plt_path += '.svg'
        print(f'Saving {plt_path}')
        plt.savefig(plt_path, format=fig_save_type)




