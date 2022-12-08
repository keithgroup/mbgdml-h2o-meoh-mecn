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

# pylint: disable=line-too-long

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mbgdml.data import PredictSet
from mbgdml.mbe import mbe_contrib
from reptar import File

save_dir = 'analysis/isomer-rankings/mbgdml'
isomer_dir = 'isomers/'
pset_dir = 'psets/'
solvents = ['h2o', 'mecn', 'meoh']  # 'h2o', 'mecn', 'meoh'
isomer_sizes = [4, 5, 6]
nbody_orders = [1, 2, 3]
save_figure = True
image_type = 'svg'

# Plot names
plot_names = {
    'h2o': '4-6h2o.temelso.etal.isomers-gdml.train1000',
    'mecn': '4-6mecn.malloum.etal.isomers-gdml.train1000',
    'meoh': '4-6meoh.boyd.etal.isomers-gdml.train1000',
}

# Reference data
isomer_data_paths = {
    'h2o': 'h2o-temelso.etal.exdir',
    'mecn': 'mecn-malloum.etal.exdir',
    'meoh': 'meoh-boyd.etal.exdir'
}
e_key = 'energy_ele_mp2.def2tzvp_orca'
g_key = 'grads_mp2.def2tzvp_orca'
e_mb_key = 'energy_ele_nbody_mp2.def2tzvp_orca'
g_mb_key = 'grads_nbody_mp2.def2tzvp_orca'

# Predicted data
pset_paths = {
    'h2o': {
        4: 'h2o/gdml/4h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
        5: 'h2o/gdml/5h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
        6: 'h2o/gdml/6h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
    },
    'mecn': {
        4: 'mecn/gdml/4mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
        5: 'mecn/gdml/5mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
        6: 'mecn/gdml/6mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
    },
    'meoh': {
        4: 'meoh/gdml/4meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
        5: 'meoh/gdml/5meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
        6: 'meoh/gdml/6meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1-gdml.train1000.npz',
    }
}

# More information: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
use_rc_params = True
font_dirs = ['../../fonts/roboto']
rc_json_path = '../matplotlib-rc-params.json'







###   SCRIPT   ###
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
data_dir = os.path.join(base_dir, 'data/')
isomer_dir = os.path.join(data_dir, isomer_dir)
pset_dir = os.path.join(data_dir, pset_dir)
save_dir = os.path.join(base_dir, save_dir)
os.makedirs(save_dir, exist_ok=True)

isomer_data_paths = {
    k:os.path.join(isomer_dir, v) for k,v in isomer_data_paths.items()
}
for k,v in pset_paths.items():
    pset_paths[k] = {
        k_iso: os.path.join(pset_dir, v_iso) for k_iso,v_iso in pset_paths[k].items()
    }

def get_data(rfile, group_key, e_key, g_key):
    """Retrieve all data used to make many-body predictions.
    """
    E = rfile.get(f'{group_key}/{e_key}')
    G = rfile.get(f'{group_key}/{g_key}')
    entity_ids = rfile.get(f'{group_key}/entity_ids')
    try:
        r_prov_ids = rfile.get(f'{group_key}/r_prov_ids')
    except Exception:
        r_prov_ids = None
    try:
        r_prov_specs = rfile.get(f'{group_key}/r_prov_specs')
    except Exception:
        r_prov_specs = None
    return E, G, entity_ids, r_prov_ids, r_prov_specs

def get_isomer_ref(rfile, isomer_size, solv_key):
    global e_key, g_key
    group_key = f'{isomer_size}{solv_key}'
    data = get_data(rfile, group_key, e_key, g_key)
    E, G = data[0], data[1]
    E *= hartree2kcalmol
    F = -G*hartree2kcalmol
    return E, F, *data[2:]

def get_isomer_mb(rfile, isomer_size, solv_key, nbody_orders):
    global e_key, g_key, e_mb_key, g_mb_key
    group_key = f'{isomer_size}{solv_key}'
    isomer_mb = get_isomer_ref(
        rfile, isomer_size, solv_key
    )
    E, F = isomer_mb[0], isomer_mb[1]
    E[:] = 0.
    F[:] = 0.

    for nbody_order in nbody_orders:
        if nbody_order == 1:
            e_key_local = e_key
            g_key_local = g_key
        else:
            e_key_local = e_mb_key
            g_key_local = g_mb_key
        group_key_mb = group_key + f'/samples_{nbody_order}{solv_key}'
        mb_data = get_data(rfile, group_key_mb, e_key_local, g_key_local)
        E_mb, G_mb = mb_data[0], mb_data[1]
        E_mb *= hartree2kcalmol
        F_mb = -G_mb*hartree2kcalmol
        mb_data= (E_mb, F_mb, *mb_data[2:])

        E, F = mbe_contrib(E, F, *isomer_mb[2:], *mb_data, operation='add')

    return E, F




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



# FIGURE #

include_ref_values = False

for solvent in solvents:
    fig, axes = plt.subplots(1, 3 , figsize=(6, 3.5), constrained_layout=True)

    data_color = mbgdml_colors[solvent]
    mbe_color = mbe_colors[solvent]
    line_width = 1.5
    marker_style = 'o'
    marker_size = 5

    rfile = File(isomer_data_paths[solvent])

    i_axis = 0
    for isomer_size in isomer_sizes:
        # Relative energies of all data are with respect to MP2 lowest.
        mp2_E, mp2_F, _, _, _ = get_isomer_ref(rfile, isomer_size, solvent)
        
        E_idx = np.argsort(mp2_E)
        mp2_E = mp2_E[E_idx]
        mp2_E_relative = mp2_E - mp2_E[0]  # Relative energies from minima.
        mp2_F = mp2_F[E_idx]


        # many-body expansion
        mbe_E, mbe_F = get_isomer_mb(rfile, isomer_size, solvent, nbody_orders)
        
        mbe_E = mbe_E[E_idx]
        mbe_E_relative = mbe_E - mp2_E[0]
        
        mbe_F = mbe_F[E_idx]


        # Predicted set.
        pset_path = pset_paths[solvent][isomer_size]
        pset = PredictSet(pset_path, Z_key='z')
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
        print(f'{solvent} {isomer_size}mer MBE MAE, MP2: {mbe_mae_mp2:.4f} kcal/mol   {mbe_f_mae_wrt_mp2:.4f} kcal/(mol A)')
        print(f'       mbGDML MAE, MP2: {mbgdml_mae_mp2:.4f} kcal/mol   {mbgdml_f_mae_wrt_mp2:.4f} kcal/(mol A)         MBE: {mbgdml_mae_mbe:.4f} kcal/mol   {mbgdml_f_mae_wrt_mbe:.4f} kcal/(mol A)')
        print()

        
        ax = axes[i_axis]

        isomer_i = range(1, len(mp2_E) + 1)

        # Axis
        if i_axis == 0:
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
        label = chr(ord('@')+(i_axis + 1))
        ax.text(
            0.071, 0.945,
            label,
            fontsize='large',
            fontweight='bold',
            transform=ax.transAxes
        )
        
        # x-axis
        ax.set_xlabel(f'{isomer_size}mer Isomers')
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

        if i_axis == 0:
            ax.legend(frameon=False, loc=(0.01, 0.78))
        
        i_axis += 1
    
    if save_figure:
        plt_path = os.path.join(save_dir, f'{plot_names[solvent]}.{image_type}')

        print(f'Saving {plot_names[solvent]}')
        plt.savefig(plt_path, dpi=1000)

    print('\n\n')