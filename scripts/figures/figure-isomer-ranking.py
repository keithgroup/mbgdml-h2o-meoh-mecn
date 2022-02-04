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
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from mbgdml.data import dataSet, predictSet
from mbgdml.utils import e_f_contribution

solvents = ['h2o', 'mecn', 'meoh']  # 'h2o', 'mecn', 'meoh'
save_figure = False
image_type = 'png'



###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Plot names
plot_names = {
    'h2o': '4-6h2o.temelso.etal.isomers-iterativetrain1000',
    'mecn': '4-6mecn.malloum.etal.isomers-iterativetrain1000',
    'meoh': '4-6meoh.boyd.etal.isomers-iterativetrain1000',
}

# Data set paths.
dset_dir = '../../data/datasets'
dset_dir_h2o = f'{dset_dir}/h2o'
dset_dir_mecn = f'{dset_dir}/mecn'
dset_dir_meoh = f'{dset_dir}/meoh'

isomer_dsets = {
    'h2o': [
        dataSet(f'{dset_dir_h2o}/4h2o/4h2o.temelso.etal-dset.npz'),
        dataSet(f'{dset_dir_h2o}/5h2o/5h2o.temelso.etal-dset.npz'),
        dataSet(f'{dset_dir_h2o}/6h2o/6h2o.temelso.etal-dset.npz')
    ],
    'mecn': [
        dataSet(f'{dset_dir_mecn}/4mecn/4mecn.malloum.etal-dset.npz'),
        dataSet(f'{dset_dir_mecn}/5mecn/5mecn.malloum.etal-dset.npz'),
        dataSet(f'{dset_dir_mecn}/6mecn/6mecn.malloum.etal-dset.npz')
    ],
    'meoh': [
        dataSet(f'{dset_dir_meoh}/4meoh/4meoh.boyd.etal-dset.npz'),
        dataSet(f'{dset_dir_meoh}/5meoh/5meoh.boyd.etal-dset.npz'),
        dataSet(f'{dset_dir_meoh}/6meoh/6meoh.boyd.etal-dset.npz')
    ]
}

isomer_mb_dsets = {
    'h2o': [
        [
            dataSet(f'{dset_dir_h2o}/1h2o/temelso.etal/4h2o.temelso.etal.dset.1h2o-dset.npz'),
            dataSet(f'{dset_dir_h2o}/2h2o/temelso.etal/4h2o.temelso.etal.dset.2h2o-dset.mb.npz'),
            dataSet(f'{dset_dir_h2o}/3h2o/temelso.etal/4h2o.temelso.etal.dset.3h2o-dset.mb.npz'),
        ],
        [
            dataSet(f'{dset_dir_h2o}/1h2o/temelso.etal/5h2o.temelso.etal.dset.1h2o-dset.npz'),
            dataSet(f'{dset_dir_h2o}/2h2o/temelso.etal/5h2o.temelso.etal.dset.2h2o-dset.mb.npz'),
            dataSet(f'{dset_dir_h2o}/3h2o/temelso.etal/5h2o.temelso.etal.dset.3h2o-dset.mb.npz'),
        ],
        [
            dataSet(f'{dset_dir_h2o}/1h2o/temelso.etal/6h2o.temelso.etal.dset.1h2o-dset.npz'),
            dataSet(f'{dset_dir_h2o}/2h2o/temelso.etal/6h2o.temelso.etal.dset.2h2o-dset.mb.npz'),
            dataSet(f'{dset_dir_h2o}/3h2o/temelso.etal/6h2o.temelso.etal.dset.3h2o-dset.mb.npz'),
        ],
    ],
    'mecn': [
        [
            dataSet(f'{dset_dir_mecn}/1mecn/malloum.etal/4mecn.malloum.etal.dset.1mecn-dset.npz'),
            dataSet(f'{dset_dir_mecn}/2mecn/malloum.etal/4mecn.malloum.etal.dset.2mecn-dset.mb.npz'),
            dataSet(f'{dset_dir_mecn}/3mecn/malloum.etal/4mecn.malloum.etal.dset.3mecn-dset.mb.npz'),
        ],
        [
            dataSet(f'{dset_dir_mecn}/1mecn/malloum.etal/5mecn.malloum.etal.dset.1mecn-dset.npz'),
            dataSet(f'{dset_dir_mecn}/2mecn/malloum.etal/5mecn.malloum.etal.dset.2mecn-dset.mb.npz'),
            dataSet(f'{dset_dir_mecn}/3mecn/malloum.etal/5mecn.malloum.etal.dset.3mecn-dset.mb.npz'),
        ],
        [
            dataSet(f'{dset_dir_mecn}/1mecn/malloum.etal/6mecn.malloum.etal.dset.1mecn-dset.npz'),
            dataSet(f'{dset_dir_mecn}/2mecn/malloum.etal/6mecn.malloum.etal.dset.2mecn-dset.mb.npz'),
            dataSet(f'{dset_dir_mecn}/3mecn/malloum.etal/6mecn.malloum.etal.dset.3mecn-dset.mb.npz'),
        ],
    ],
    'meoh': [
        [
            dataSet(f'{dset_dir_meoh}/1meoh/boyd.etal/4meoh.boyd.etal.dset.1meoh-dset.npz'),
            dataSet(f'{dset_dir_meoh}/2meoh/boyd.etal/4meoh.boyd.etal.dset.2meoh-dset.mb.npz'),
            dataSet(f'{dset_dir_meoh}/3meoh/boyd.etal/4meoh.boyd.etal.dset.3meoh-dset.mb.npz'),
        ],
        [
            dataSet(f'{dset_dir_meoh}/1meoh/boyd.etal/5meoh.boyd.etal.dset.1meoh-dset.npz'),
            dataSet(f'{dset_dir_meoh}/2meoh/boyd.etal/5meoh.boyd.etal.dset.2meoh-dset.mb.npz'),
            dataSet(f'{dset_dir_meoh}/3meoh/boyd.etal/5meoh.boyd.etal.dset.3meoh-dset.mb.npz'),
        ],
        [
            dataSet(f'{dset_dir_meoh}/1meoh/boyd.etal/6meoh.boyd.etal.dset.1meoh-dset.npz'),
            dataSet(f'{dset_dir_meoh}/2meoh/boyd.etal/6meoh.boyd.etal.dset.2meoh-dset.mb.npz'),
            dataSet(f'{dset_dir_meoh}/3meoh/boyd.etal/6meoh.boyd.etal.dset.3meoh-dset.mb.npz'),
        ],
    ]
}

# pset paths.
pset_dir = '../../data/predictsets'
pset_dir_h2o = f'{pset_dir}/h2o'
pset_dir_mecn = f'{pset_dir}/mecn'
pset_dir_meoh = f'{pset_dir}/meoh'

pset_paths = {
    'h2o': [
        predictSet(f'{pset_dir_h2o}/4h2o/4h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_h2o}/5h2o/5h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_h2o}/6h2o/6h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz')
    ],
    'mecn': [
        predictSet(f'{pset_dir_mecn}/4mecn/4mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_mecn}/5mecn/5mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_mecn}/6mecn/6mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz')
    ],
    'meoh': [
        predictSet(f'{pset_dir_meoh}/4meoh/4meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_meoh}/5meoh/5meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_meoh}/6meoh/6meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz')
    ]
}

save_dir = '../../analysis/isomer-predictions'

if save_dir[-1] != '/':
    save_dir += '/'

mbgdml_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}
mbe_colors = {
    'h2o': '#b6e3fa',
    'mecn': '#bfe5da',
    'meoh': '#ffe1e3',
}
ref_color = 'silver'


# FIGURE #

# Setting up general figure properties
font = {'family' : 'sans-serif',
        'size'   : 8}
plt.rc('font', **font)

include_ref_values = False

for solvent in solvents:
    fig, axes = plt.subplots(1, 3 , figsize=(6, 3.5), constrained_layout=True)

    data_color = mbgdml_colors[solvent]
    mbe_color = mbe_colors[solvent]
    line_width = 1.5
    marker_style = 'o'
    marker_size = 5

    for i in range(len(isomer_dsets[solvent])):
        # Relative energies of all data are with respect to MP2 lowest.
        nmer_dset = isomer_dsets[solvent][i]
        mp2_E = nmer_dset.E
        E_idx = np.argsort(mp2_E)
        mp2_E = mp2_E[E_idx]
        mp2_E_relative = mp2_E - mp2_E[0]  # Relative energies from minima.
        mp2_F = nmer_dset.F[E_idx]

        # many-body expansion
        all_mbe_dsets = isomer_mb_dsets[solvent][i]  # 1-, 2-, and 3-body dsets

        mb_dset = copy(nmer_dset)
        mb_dset.E = np.zeros(mb_dset.E.shape)
        mb_dset.F = np.zeros(mb_dset.F.shape)
        mb_dset = e_f_contribution(mb_dset, all_mbe_dsets, 'add')
        
        mbe_E = mb_dset.E
        mbe_E = mbe_E[E_idx]
        mbe_E_relative = mbe_E - mp2_E[0]
        
        mbe_F = mb_dset.F[E_idx]

        # Predicted set.
        pset = pset_paths[solvent][i]
        mbgdml_E, mbgdml_F = pset.nbody_predictions([1, 2, 3])
        mbgdml_E = mbgdml_E[E_idx]
        mbgdml_E_relative = mbgdml_E - mp2_E[0]

        mbgdml_F = mbgdml_F[E_idx]
        
        # Prints model performance information.
        mbe_mae_mp2 = np.mean(np.abs(mbe_E - mp2_E))
        mbe_f_mae_wrt_mp2 = np.mean(np.abs(mbe_F.flatten() - mp2_F.flatten()))
        mbgdml_mae_mp2 = np.mean(np.abs(mbgdml_E - mp2_E))
        mbgdml_mae_mbe = np.mean(np.abs(mbgdml_E - mbe_E))
        mbgdml_f_mae_wrt_mbe = np.mean(np.abs(mbgdml_F.flatten() - mbe_F.flatten()))
        mbgdml_f_mae_wrt_mp2 = np.mean(np.abs(mbgdml_F.flatten() - mp2_F.flatten()))
        print(f'{solvent} {i+4}mer MBE MAE, MP2: {mbe_mae_mp2:.4f} kcal/mol   {mbe_f_mae_wrt_mp2:.4f} kcal/(mol A)')
        print(f'       mbGDML MAE, MP2: {mbgdml_mae_mp2:.4f} kcal/mol   {mbgdml_f_mae_wrt_mp2:.4f} kcal/(mol A)         MBE: {mbgdml_mae_mbe:.4f} kcal/mol   {mbgdml_f_mae_wrt_mbe:.4f} kcal/(mol A)')
        print()

        
        ax = axes[i]

        isomer_i = range(1, len(mp2_E) + 1)

        # Axis
        if i == 0:
            ax.set_ylabel('$\Delta$E (kcal/mol)')

        # MP2 reference data
        ax.plot(
            isomer_i,
            mp2_E_relative,
            color=ref_color,
            label='MP2',
            marker='s',
            markersize=marker_size,
            markeredgewidth=0,
            linestyle='--',
            linewidth=line_width,
            zorder=0,
            alpha=1
        )

        ax.plot(
            isomer_i,
            mbe_E_relative,
            color=mbe_color,
            label='MBE',
            marker='s',
            markersize=marker_size,
            markeredgewidth=0,
            linestyle='--',
            linewidth=line_width,
            zorder=1,
            alpha=1
        )

        ax.plot(
            isomer_i,
            mbgdml_E_relative,
            color=data_color,
            label='mbGDML',
            marker=marker_style,
            markersize=marker_size,
            markeredgewidth=0,
            linestyle='-',
            linewidth=line_width,
            zorder=2,
            alpha=1
        )
        
        # Subplot label
        label = chr(ord('@')+(i + 1))
        ax.text(
            0.071, 0.945,
            label,
            fontsize='large',
            fontweight='bold',
            transform=ax.transAxes
        )
        
        # x-axis
        ax.set_xlabel(f'{i+4}mer Isomers')
        ax.set_xlim(xmin=1)
        ax.set_xticks([])

        """
        ax.set_xticks(np.arange(1, len(mp2_E) + 1, 1))
        ax.set_xlim(xmin=1)
        xtick_labels = []
        j = 1
        while j <= len(mp2_E):
            if len(mp2_E) < 10:
                xtick_labels.append(f'{j}')
            else:
                if j%2 == 0:
                    xtick_labels.append('')
                else:
                    xtick_labels.append(f'{j}')
            j += 1
        ax.set_xticklabels(xtick_labels)
        """

        # y-axis
        y_tick_start, y_tick_end = ax.get_ylim()
        set_0_min = True
        all_data = [mp2_E_relative, mbe_E_relative, mbgdml_E_relative]
        for data in all_data:
            e_min = np.min(data)
            if -0.5 <= e_min < 0.0:
                set_0_min = False
                y_major_start = 0
                y_minor_start = -0.5
            elif -1.0 <= e_min < -0.5:
                set_0_min = False
                y_major_start = -1
                y_minor_start = -0.5
                break
            elif -1.5 <= e_min < -1.0:
                set_0_min = False
                y_major_start = -1
                y_minor_start = -0.5
                break
            else:
                y_major_start = 0
                y_minor_start = 0.5
        
        if y_tick_end > 10:
            y_tick_major_step = 2
            y_tick_minor_step = 1
            y_minor_start *= 2
        else:
            y_tick_major_step = 1
            y_tick_minor_step = 0.5
        
        ax.set_yticks(np.arange(y_major_start, y_tick_end, y_tick_major_step))
        ax.set_yticks(np.arange(y_minor_start, y_tick_end, y_tick_minor_step), minor=True)

        if set_0_min:
            ax.set_ylim(ymin=0)
        

        if i == 0:
            ax.legend(frameon=False, loc=(0.01, 0.78))
    
    if save_figure:
        plt_path = f'{save_dir}{plot_names[solvent]}.{image_type}'

        print(f'Saving {plot_names[solvent]}')
        plt.savefig(plt_path, dpi=1000)

    print('\n\n')