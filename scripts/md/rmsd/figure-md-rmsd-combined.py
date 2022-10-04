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

"""Plots the RMSDs of hexamer MD simulations for all solvents."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import rmsd

# initial - RMSD with respect to the initial structure of the same MD simulation
# compare - Choose a reference to compute RMSD at each time step to. Reference is not included in figure.
# compare is a superior analysis
rmsd_type = 'initial'
rmsd_ref_idx = 0

fig_save_types = ['svg', 'eps']  # eps or png
figsize = (6, 3.2)
line_width = 1.5

save = True
plot_name = '6mer-ase.md-rmsd-all'
save_dir = 'analysis/md/6mer/rmsd'

md_npz_paths_all ={
    'h2o': [
        ['data/md/h2o/6h2o/orca/6h2o-ase.md-orca.mp2.def2tzvp.300k-0/6h2o-ase.md-orca.mp2.def2tzvp.300k-0.npz', 'data/md/h2o/6h2o/orca/6h2o-ase.md-orca.mp2.def2tzvp.300k-0.r/6h2o-ase.md-orca.mp2.def2tzvp.300k-0.r.npz'],
        'data/md/h2o/6h2o/orca/6h2o-ase.md-orca.rimp2.def2tzvp.300k-0/6h2o-ase.md-orca.rimp2.def2tzvp.300k-0.npz',
        'data/md/h2o/6h2o/orca/6h2o-ase.md-orca.mp2.def2svp.300k-0/6h2o-ase.md-orca.mp2.def2svp.300k-0.npz',
        'data/md/h2o/6h2o/gdml/6h2o-ase.md-mbgdml.train1000.300k-0/6h2o-ase.md-mbgdml.train1000.300k-0.npz',
        'data/md/h2o/6h2o/gap/6h2o-ase.md-gap.train1000.300k-0/6h2o-ase.md-gap.train1000.300k-0.npz',
        'data/md/h2o/6h2o/schnet/6h2o-ase.md-schnet.train1000.300k-0/6h2o-ase.md-schnet.train1000.300k-0.npz',
        'data/md/h2o/6h2o/gfn2/6h2o-ase.md-gfn2.300k-0/6h2o-ase.md-gfn2.300k-0.npz',
    ],
    'mecn': [
        ['data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-0/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-0.npz', 'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-1/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-1.npz', 'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-2/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-2.npz', 'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-3/6mecn-ase.md-orca.mp2.def2tzvp.300k-0-3.npz',],
        ['data/md/mecn/6mecn/orca/6mecn-ase.md-orca.rimp2.def2tzvp.300k-0/6mecn-ase.md-orca.rimp2.def2tzvp.300k-0.npz', 'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.rimp2.def2tzvp.300k-0-1/6mecn-ase.md-orca.rimp2.def2tzvp.300k-0-1.npz',],
        'data/md/mecn/6mecn/orca/6mecn-ase.md-orca.mp2.def2svp.300k-0-0/6mecn-ase.md-orca.mp2.def2svp.300k-0-0.npz',
        'data/md/mecn/6mecn/gdml/6mecn-ase.md-gdml.train1000.300k-0/6mecn-ase.md-mbgdml.train1000.300k-0.npz',
        'data/md/mecn/6mecn/gap/6mecn-ase.md-gap.train1000.300k-0/6mecn-ase.md-gap.train1000.300k-0.npz',
        'data/md/mecn/6mecn/schnet/6mecn-ase.md-schnet.train1000.300k-0/6mecn-ase.md-schnet.train1000.300k-0.npz',
        'data/md/mecn/6mecn/gfn2/6mecn-ase.md-gfn2.300k-0/6mecn-ase.md-gfn2.300k-0.npz',
    ],
    'meoh': [
        ['data/md/meoh/6meoh/orca/6meoh-ase.md-orca.mp2.def2tzvp.300k-0/6meoh-ase.md-orca.mp2.def2tzvp.300k-0.npz', 'data/md/meoh/6meoh/orca/6meoh-ase.md-orca.mp2.def2tzvp.300k-0.r/6meoh-ase.md-orca.mp2.def2tzvp.300k-0.r.npz'],
        'data/md/meoh/6meoh/orca/6meoh-ase.md-orca.rimp2.def2tzvp.300k-0/6meoh-ase.md-orca.rimp2.def2tzvp.300k-0.npz',
        'data/md/meoh/6meoh/orca/6meoh-ase.md-orca.mp2.def2svp.300k-0/6meoh-ase.md-orca.mp2.def2svp.300k-0.npz',
        'data/md/meoh/6meoh/gdml/6meoh-ase.md-gdml.train1000.300k-0/6meoh-ase.md-mbgdml.train1000.300k-0.npz',
        'data/md/meoh/6meoh/gap/6meoh-ase.md-gap.train1000.300k-0/6meoh-ase.md-gap.train1000.300k-0.npz',
        'data/md/meoh/6meoh/schnet/6meoh-ase.md-schnet.train1000.300k-0/6meoh-ase.md-schnet.train1000.300k-0.npz',
        'data/md/meoh/6meoh/gfn2/6meoh-ase.md-gfn2.300k-0/6meoh-ase.md-gfn2.300k-0.npz',
    ]
}
labels = ['MP2', 'RI-MP2', 'def2-SVP', 'mbGDML', 'mbGAP', 'mbSchNet', 'GFN2-xTB']
colors_all = {
    'h2o':  ['#343a40', '#6c757d', '#ced4da', '#03045e', '#0077b6', '#90e0ef', '#FFE7BE'],
    'mecn': ['#343a40', '#6c757d', '#ced4da', '#1b4332', '#2d6a4f', '#95d5b2', '#FFE7BE'],
    'meoh': ['#343a40', '#6c757d', '#ced4da', '#590d22', '#a4133c', '#ff8fa3', '#FFE7BE'],
}

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


# Compute RMSE of MD simulation with respect to the initial structure.

def calc_rmsd(R_ref, R):
    """Calculates the RMSD between two structures.

    Parameters
    ----------
    R_ref : :obj:`numpy.ndarray`
        A 2D array of Cartesian coordinates.
    R : :obj:`numpy.ndarray`
        A 2D array of Cartesian coordinates.
    
    Returns
    -------
    :obj:`float`
        The RMSD of two structures.
    """
    N = R_ref.shape[0]
    assert R_ref.ndim == 2 and R.ndim == 2
    A = R_ref
    B = R
    A -= rmsd.centroid(A)
    B -= rmsd.centroid(B)
    U = rmsd.kabsch(A, B)
    A = np.dot(A, U)
    return rmsd.rmsd(A, B)

        


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

fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True, sharey=True)

ax_idx = 0
for solv_key in md_npz_paths_all.keys():
    print(f'Working on {solv_key}')
    md_npz_paths = md_npz_paths_all[solv_key]
    colors = colors_all[solv_key]
    ax = axes[ax_idx]

    # Collects (and possibly combines) all Cartesian coordinates.
    md_data = []
    for i in range(len(md_npz_paths)):
        md_paths = md_npz_paths[i]
        # Single path, no splits in MD trajectory.
        if isinstance(md_paths, str):
            md_paths = [os.path.join(base_dir, md_paths)]
        elif isinstance(md_paths, list):
            md_paths = [os.path.join(base_dir, j) for j in md_paths]

        md_R = [
            dict(np.load(md_path, allow_pickle=True))['R'] for md_path in md_paths
        ]
        # For restarted MD simulations the last structure is also the initial
        # structure of the restarted MD simulation. We need to remove this structure
        # from the restarts.
        if len(md_R) > 1:
            for i in range(1, len(md_R)):
                md_R[i] = md_R[i][1:]

        md_R = np.array(md_R, dtype=object)
        md_R = np.vstack(md_R).astype(np.double)
        md_data.append(md_R)

    rmsd_data = []
    if rmsd_type == 'initial':
        R_ref_idx = 0
        for R in md_data:
            rmsd_md = np.zeros(R.shape[0], dtype=np.double)
            for i in range(len(rmsd_md)):
                rmsd_md[i] = calc_rmsd(R[R_ref_idx], R[i])
            rmsd_data.append(rmsd_md)
    elif rmsd_type == 'compare':
        for i in range(len(md_data)):
            # Skip the reference RMSE
            if i == rmsd_ref_idx:
                rmsd_data.append(None)
                continue
            R = md_data[i]
            rmsd_md = np.zeros(R.shape[0], dtype=np.double)
            for j in range(len(rmsd_md)):
                rmsd_md[j] = calc_rmsd(md_data[rmsd_ref_idx][j], R[j])
            rmsd_data.append(rmsd_md)

    for i in range(len(md_data)):
        if rmsd_type == 'compare' and i == rmsd_ref_idx:
            continue
        rmsd_plot = rmsd_data[i]
        label_plot = labels[i]
        color_plot = colors[i]

        if label_plot == '':
            label_plot = None
        ax.plot(
            rmsd_plot,
            color=color_plot,
            marker='', markersize=0, 
            linestyle='-', linewidth=line_width,
            label=label_plot,
            alpha=1.0,
            zorder=-i
        )

    # Subplot label
    label = chr(ord('@')+(ax_idx + 1))
    ax.text(
        0.02, 0.955,
        label,
        fontsize='large',
        fontweight='bold',
        transform=ax.transAxes
    )
    
    ax.set_xlim(xmin=0)
    
    ax_idx += 1

# X axis
axes[1].set_xlabel('Time (fs)')

# Y axis
axes[0].set_ylabel('RMSD (Ang.)')
axes[0].set_ylim(ymin=0)

axes[0].legend(frameon=False)

if save:
    plt_path = os.path.join(save_dir, plot_name + '-' + rmsd_type)

    print(f'Saving {plt_path}')
    for fig_save_type in fig_save_types:
        plt.savefig(plt_path + f'.{fig_save_type}', format=fig_save_type, dpi=1200)

