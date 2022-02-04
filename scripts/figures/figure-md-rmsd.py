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

import numpy as np
import matplotlib.pyplot as plt

fig_save_type = 'svg'  # eps or png
figsize = (6, 6)
font_size = 18  # 8

# MD directory information
md_dir = '../../data/md'
md_dir_h2o = f'{md_dir}/h2o'
md_dir_mecn = f'{md_dir}/mecn'
md_dir_meoh = f'{md_dir}/meoh'

# H2O
"""
md_dir_solvent = md_dir_h2o
solvent = 'h2o'
md_ref_path = f'{md_dir_solvent}/6h2o/6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k/6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k.npz'

md_paths = {
    'mbgdml3.randomtrain200': f'{md_dir_solvent}/6h2o/6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-mbgdml3.randomtrain200.300k/6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-mbgdml3.randomtrain200.300k.npz',
    'mbgdml3.iterativetrain500': f'{md_dir_solvent}/6h2o/6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain500.300k/6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain500.300k.npz',
    'mbgdml3.iterativetrain1000': f'{md_dir_solvent}/6h2o/6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain1000.300k/6h2o.temelso.etal.pr.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain1000.300k.npz',
}
labels = {
    'mbgdml3.randomtrain200':     'mbGDML @   200',
    'mbgdml3.iterativetrain500':  'mbGDML @   500',
    'mbgdml3.iterativetrain1000': 'mbGDML @ 1000'
}
key_order = [
    'mbgdml3.randomtrain200', 'mbgdml3.iterativetrain500', 'mbgdml3.iterativetrain1000'
]
key_alphas = {
    'mbgdml3.randomtrain200': 0.25,
    'mbgdml3.iterativetrain500': 0.5,
    'mbgdml3.iterativetrain1000': 1.0
}
plot_name = '6h2o.temelso.etal.pr.md.gfn2.300k.step10000.ase.md-rmsd'
"""


# MeCN
"""
solvent = 'mecn'
md_dir_solvent = md_dir_mecn
md_ref_path = f'{md_dir_solvent}/6mecn/6mecn.malloum.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k/6mecn.malloum.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k.npz'
md_paths = {
    'mbgdml3.randomtrain200': f'{md_dir_solvent}/6mecn/6mecn.malloum.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.randomtrain200.300k/6mecn.malloum.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.randomtrain200.300k.npz',
    'mbgdml3.iterativetrain500': f'{md_dir_solvent}/6mecn/6mecn.malloum.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain500.300k/6mecn.malloum.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain500.300k.npz',
    'mbgdml3.iterativetrain1000': f'{md_dir_solvent}/6mecn/6mecn.malloum.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain1000.300k/6mecn.malloum.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain1000.300k.npz',
}
labels = {
    'mbgdml3.randomtrain200':     'mbGDML @   200',
    'mbgdml3.iterativetrain500':  'mbGDML @   500',
    'mbgdml3.iterativetrain1000': 'mbGDML @ 1000'
}
key_order = [
    'mbgdml3.randomtrain200', 'mbgdml3.iterativetrain500', 'mbgdml3.iterativetrain1000'
]
key_alphas = {
    'mbgdml3.randomtrain200': 0.25,
    'mbgdml3.iterativetrain500': 0.5,
    'mbgdml3.iterativetrain1000': 1.0
}
plot_name = '6mecn.malloum.etal.1.md.gfn2.300k.step10000.ase.md-rmsd'
"""


# MeOH

solvent = 'meoh'
md_dir_solvent = md_dir_meoh
md_ref_path = f'{md_dir_solvent}/6meoh/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-orca.mp2.def2tzvp.300k.npz'
md_paths = {
    'mbgdml3.randomtrain200': f'{md_dir_solvent}/6meoh/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.randomtrain200.300k/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.randomtrain200.300k.npz',
    'mbgdml3.iterativetrain500': f'{md_dir_solvent}/6meoh/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain500.300k/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain500.300k.npz',
    'mbgdml3.iterativetrain1000': f'{md_dir_solvent}/6meoh/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain1000.300k/6meoh.boyd.etal.1.md.gfn2.300k.md.gfn2.300k.step10000-ase.md-mbgdml3.iterativetrain1000.300k.npz',
}
labels = {
    'mbgdml3.randomtrain200':     'mbGDML @   200',
    'mbgdml3.iterativetrain500':  'mbGDML @   500',
    'mbgdml3.iterativetrain1000': 'mbGDML @ 1000'
}
key_order = [
    'mbgdml3.randomtrain200', 'mbgdml3.iterativetrain500', 'mbgdml3.iterativetrain1000'
]
key_alphas = {
    'mbgdml3.randomtrain200': 0.25,
    'mbgdml3.iterativetrain500': 0.5,
    'mbgdml3.iterativetrain1000': 1.0
}
plot_name = '6meoh.boyd.etal.1.md.gfn2.300k.step10000.ase.md-rmsd'



save = False
save_dir = '../../analysis/md/md-rmsd'


# Plot

solvent_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}
cluster_num_color = 'silver'






###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if save_dir[-1] != '/':
    save_dir += '/'

# Loads MD data
data_ref = dict(np.load(md_ref_path, allow_pickle=True))
data = {}
for key in md_paths.keys():
    data[key] = dict(np.load(md_paths[key], allow_pickle=True))

# Gets coordinate data
R_ref = data_ref['R']

# RMSD function
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
    return np.sqrt(((R_ref-R) ** 2).sum() * (1/N))

# Calculates RMSD of every structure with respect to the reference structure at
# the same step.
"""
rmsd_ref = np.zeros(len(R_ref))
for i in range(len(R_ref)):
    rmsd_ref[i] = calc_rmsd(R_ref[0], R_ref[i])  # Wrt the structure at same step
"""

rmsd_all = {}
for key in data.keys():
    R = data[key]['R']
    rmsd_all[key] = np.zeros(len(R_ref))
    for i in range(len(rmsd_all[key])):
        rmsd_all[key][i] = calc_rmsd(R_ref[i], R[i])  # Wrt the structure at same step
        """
        rmsd_all[key][i] = calc_rmsd(R_ref[0], R[i])  # Wrt the structure at same step
        """

print('Final RMSD values')
for k,v in rmsd_all.items():
    print(f'{k}: {v[-1]:.3f}')

###   FIGURE   ###

# Setting up general figure properties
font = {'family' : 'sans-serif',
        'size'   : font_size}
plt.rc('font', **font)

fig, ax = plt.subplots(1, 1 , figsize=figsize, constrained_layout=True)

"""
# Ref RMSD
ax.plot(
    rmsd_ref,
    color='dimgrey',
    marker='', markersize=0, 
    linestyle='-', linewidth=1.5,
    label='MP2/def2-TZVP',
    alpha=1.0
)
"""

# mbGDML RMSD
for key in key_order:
    rmsd = rmsd_all[key]
    label = labels[key]
    alpha = key_alphas[key]
    ax.plot(
        rmsd,
        color=solvent_colors[solvent],
        marker='', markersize=0, 
        linestyle='-', linewidth=1.5,
        label=label,
        alpha=alpha
    )

# X axis
ax.set_xlabel('Time (fs)')
ax.set_xlim(0, len(R_ref))

# Y axis
ax.set_ylabel('RMSD ($\AA$)')

ax.legend(frameon=False)

if save:
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]
    plt_path = f'{save_dir}/{plot_name}'
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



