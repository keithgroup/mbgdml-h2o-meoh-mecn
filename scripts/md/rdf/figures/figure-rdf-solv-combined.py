#!/usr/bin/env python3

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

"""Performs a path similarity analysis with MDAnalysis."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

rdf_info = {
    'h2o': {
        'npz_path': 'analysis/md/rdf/h2o/137h2o-mbgdml-nvt_1_2-rdf-oo.npz',
        'exp_path': 'external/md/h2o-rdf/soper2013radial.csv',
        'exp_csv_label': 'goo',
        'exp_label': 'Soper',
        'solv_label': 'H$\mathregular{_2}$O',
        'r_max': 8.0,
        'mbml_color': '#4ABBF3',
        'plot_ylabel': 'g$\mathregular{_{OO}}$(r)',
    },
    'mecn': {
        'npz_path': 'analysis/md/rdf/mecn/67mecn-mbgdml-nvt_1_2_3-298-rdf-nn.npz',
        'exp_path': 'external/md/mecn-rdf/hernandez2020general-fig7-nn.csv',
        'exp_csv_label': 'g',
        'exp_label': 'Hernández-Cobos et al.',
        'solv_label': 'MeCN',
        'r_max': 9.0,
        'mbml_color': '#61BFA3',
        'plot_ylabel': 'g$\mathregular{_{NN}}$(r)',
    },
    'meoh': {
        'npz_path': 'analysis/md/rdf/meoh/61meoh-mbgdml-nvt_1_2_3-rdf-oo.npz',
        'exp_path': 'external/md/meoh-rdf/yamaguchi1999structure-fig6-oo.csv',
        'exp_csv_label': 'g',
        'exp_label': 'Yamaguchi et al.',
        'solv_label': 'MeOH',
        'r_max': 8.0,
        'mbml_color': '#FFB5BA',
        'plot_ylabel': 'g$\mathregular{_{OO}}$(r)',
    },
}
keys_order = ['h2o', 'mecn', 'meoh']

ref_color = '#6c757d'
plot_xlabel = 'r (Ang.)'
linewidth = 1.5

save_path = 'analysis/md/rdf/solvent-rdf-combined'
fig_types = ['svg', 'eps']
figsize = (3.2, 6)

# More information: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
use_rc_params = True
font_dirs = ['../../../../fonts/roboto']
rc_json_path = '../../../matplotlib-rc-params.json'





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../../'
data_dir = os.path.join(base_dir, 'data/')
md_dir = os.path.join(data_dir, 'md/')
save_path = os.path.join(base_dir, save_path)



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

fig, axes = plt.subplots(len(keys_order), 1, constrained_layout=True, sharex=True, figsize=figsize)

for i in range(len(keys_order)):
    solv_key = keys_order[i]
    solv_info = rdf_info[keys_order[i]]

    ax = axes[i]

    npz_path = os.path.join(base_dir, solv_info['npz_path'])
    exp_path = os.path.join(data_dir, solv_info['exp_path'])

    # Get data
    md_data = dict(np.load(npz_path, allow_pickle=True))
    r_md = md_data['bins']
    g_md = md_data['rdf']

    # Trim MD data
    md_idxs = np.argwhere(r_md <= solv_info['r_max']).T[0]
    r_md = r_md[md_idxs]
    g_md = g_md[md_idxs]

    # Get experimental data
    df = pd.read_csv(exp_path)
    r_exp = df['r'].values
    g_exp = df[solv_info['exp_csv_label']].values

    ax.plot(
        r_md, g_md, label=solv_info['solv_label'], zorder=0,
        linestyle='-', color=solv_info['mbml_color'], linewidth=linewidth
    )
    ax.plot(
        r_exp, g_exp, label=None, zorder=1,
        linestyle=(0, (5, 4)), color=ref_color, linewidth=linewidth
    )
    ax.axhline(1.0, zorder=-1, alpha=1.0, color='silver', linestyle=(0, (1, 4)))

    ax.legend(loc='lower right', frameon=False)

    ax.set_xlim(0, solv_info['r_max'])

    ax.set_ylabel(solv_info['plot_ylabel'])
    ax.set_ylim(ymin=0)

    ax.set_ylim(bottom=0.0)
    y_ticks = ax.get_yticks(minor=False)
    y_tick_max = np.true_divide(np.floor(np.max(y_ticks) * 10), 10)
    y_tick_major_step = 1.0
    y_tick_minor_step = 0.5
    ax.set_yticks(np.arange(start=0.0, stop=y_tick_max, step=y_tick_major_step))
    ax.set_yticks(np.arange(0.0+y_tick_minor_step, y_tick_max, y_tick_minor_step), minor=True)

    # Subplot label
    label = chr(ord('@')+(i + 1))
    ax.text(
        0.02, 0.90,
        label,
        fontsize='large',
        fontweight='bold',
        transform=ax.transAxes
    )

axes[-1].set_xlabel('r (Ang.)')

for fig_type in fig_types:
    plt.savefig(save_path + f'.{fig_type}', dpi=1000)
