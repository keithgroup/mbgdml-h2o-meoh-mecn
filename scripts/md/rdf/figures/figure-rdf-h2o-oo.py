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

"""Plot O-O site RDF curve of water."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

npz_path = 'analysis/md/rdf/h2o/137h2o-mbgdml-nvt_1_2-rdf-oo.npz'
exp_path = 'external/md/h2o-rdf/soper2013radial.csv'
exp_label = 'Soper'

x_max = 8.0

mbml_color = '#4ABBF3'
ref_color = '#6c757d'

save_path = 'analysis/md/rdf/h2o/137h2o-mbgdml-nvt_1_2-rdf-oo'
fig_types = ['svg', 'eps']
fig_size = (3.2, 3.2)

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

npz_path = os.path.join(base_dir, npz_path)
exp_path = os.path.join(data_dir, exp_path)
save_path = os.path.join(base_dir, save_path)


# Get data
md_data = dict(np.load(npz_path, allow_pickle=True))

r_md = md_data['bins']
g_md = md_data['rdf']

# Trim MD data
md_idxs = np.argwhere(r_md <= x_max).T[0]
r_md = r_md[md_idxs]
g_md = g_md[md_idxs]

# Get experimental data
df = pd.read_csv(exp_path)
r_exp = df['r'].values
g_exp = df['goo'].values

# Find peak
peak_md = np.argmax(g_md)
peak_exp = np.argmax(g_exp)
peak_md_r = r_md[peak_md]
peak_exp_r = r_exp[peak_exp]
print(f'MD peak:   {g_md[peak_md]:.2f}')
print(f'Exp. peak: {g_exp[peak_exp]:.2f}')
print(f'Difference: {g_md[peak_md]-g_exp[peak_exp]:.2f}')

print(f'\nMD r_peak:   {peak_md_r:.2f} Ang')
print(f'Exp. r_peak: {peak_exp_r:.2f} Ang')
print(f'Difference: {peak_md_r-peak_exp_r:.2f} Ang')


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

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=fig_size)

ax.plot(
    r_md, g_md, label='mbGDML', zorder=0,
    linestyle='-', color=mbml_color, linewidth=1.5
)
ax.plot(
    r_exp, g_exp, label=exp_label, zorder=1,
    linestyle=(0, (5, 4)), color=ref_color, linewidth=1.5
)
ax.axhline(1.0, zorder=-1, alpha=1.0, color='silver', linestyle=(0, (1, 4)))

ax.set_xlabel(r'r $\left( \mathbf{\AA} \right)$')
ax.set_xlim(0, x_max)

ax.set_ylabel('g$\mathregular{_{OO}}$(r)')
ax.set_ylim(ymin=0)

plt.legend(frameon=False)
for fig_type in fig_types:
    plt.savefig(save_path + f'.{fig_type}', dpi=1000)
