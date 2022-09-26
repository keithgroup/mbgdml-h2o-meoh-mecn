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

import matplotlib.pyplot as plt
import numpy as np
import os
import rmsd

# initial - RMSd with respect to the initial structure of the same MD simulation
# compare - Choose a reference to compute RMSD at each time step to. Reference is not included in figure.
rmsd_type = 'compare'
rmsd_ref_idx = 0

fig_save_type = 'svg'  # eps or png
figsize = (6, 6)
font_size = 18  # 8
line_width = 2.5

save = True
plot_name = '6meoh-ase.md-rmsd'
save_dir = 'analysis/md/6mer/rmsd'

md_npz_paths = [
    ['data/md/meoh/6meoh/orca/6meoh-ase.md-orca.mp2.def2tzvp.300k-0/6meoh-ase.md-orca.mp2.def2tzvp.300k-0.npz', 'data/md/meoh/6meoh/orca/6meoh-ase.md-orca.mp2.def2tzvp.300k-0.r/6meoh-ase.md-orca.mp2.def2tzvp.300k-0.r.npz'],
    'data/md/meoh/6meoh/orca/6meoh-ase.md-orca.rimp2.def2tzvp.300k-0/6meoh-ase.md-orca.rimp2.def2tzvp.300k-0.npz',
    'data/md/meoh/6meoh/orca/6meoh-ase.md-orca.mp2.def2svp.300k-0/6meoh-ase.md-orca.mp2.def2svp.300k-0.npz',
    'data/md/meoh/6meoh/gdml/6meoh-ase.md-gdml.train1000.300k-0/6meoh-ase.md-mbgdml.train1000.300k-0.npz',
    'data/md/meoh/6meoh/gap/6meoh-ase.md-gap.train1000.300k-0/6meoh-ase.md-gap.train1000.300k-0.npz',
    'data/md/meoh/6meoh/schnet/6meoh-ase.md-schnet.train1000.300k-0/6meoh-ase.md-schnet.train1000.300k-0.npz',
    'data/md/meoh/6meoh/gfn2/6meoh-ase.md-gfn2.300k-0/6meoh-ase.md-gfn2.300k-0.npz',
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
    None,  # MP2 is never plotted
    '#6c757d',
    '#ced4da',
    '#590d22',
    '#a4133c',
    '#ff4d6d',
    '#ff8fa3',
    '#ffccd5',
]

###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
save_dir = os.path.join(base_dir, save_dir)

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
        The RMSD of two structures (unoptimized).
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
    # return np.sqrt(((R_ref-R) ** 2).sum() * (1/N))

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

        


# Plotting figure
# Setting up general figure properties
font = {'family' : 'sans-serif',
        'size'   : font_size}
plt.rc('font', **font)

fig, ax = plt.subplots(1, 1 , figsize=figsize, constrained_layout=True)

for i in range(len(md_data)):
    if rmsd_type == 'compare' and i == rmsd_ref_idx:
        continue
    rmsd = rmsd_data[i]
    label = labels[i]
    color = colors[i]

    if label == '':
        label = None
    ax.plot(
        rmsd,
        color=color,
        marker='', markersize=0, 
        linestyle='-', linewidth=line_width,
        label=label,
        alpha=1.0
    )

# X axis
ax.set_xlabel('Time (fs)')
ax.set_xlim(xmin=0)

# Y axis
ax.set_ylabel('RMSD ($\AA$)')
ax.set_ylim(ymin=0)

ax.legend(frameon=False)

if save:
    plt_path = os.path.join(save_dir, plot_name + '-' + rmsd_type)

    print(f'Saving {plt_path}')
    if fig_save_type == 'png':
        plt_path += '.png'
        plt.savefig(plt_path, format=fig_save_type, dpi=1200)
    elif fig_save_type == 'eps':
        plt_path += '.eps'
        plt.savefig(plt_path, format=fig_save_type, dpi=1200)
    elif fig_save_type == 'svg':
        plt_path += '.svg'
        plt.savefig(plt_path, format=fig_save_type)




