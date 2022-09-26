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

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

npz_path = 'analysis/md/rdf/mecn/67mecn-mbgdml-nvt_1_2_3-298-rdf-c_n-c_n.npz'  # No file extension
exp_path = 'external/md/mecn-rdf/hernandez2020general-fig7-cncn.csv'
exp_label = 'Hernández-Cobos et al.'

r_max = 9

save_path = f'analysis/md/rdf/mecn/67mecn-mbgdml-nvt_1_2_3-298-rdf-c_n-c_n.png'


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
md_idxs = np.argwhere(r_md <= r_max).T[0]
r_md = r_md[md_idxs]
g_md = g_md[md_idxs]

df = pd.read_csv(exp_path)
r_exp = df['r'].values
g_exp = df['g'].values

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

fig, ax = plt.subplots(1, 1, constrained_layout=True)

ax.plot(
    r_md, g_md, label='mbGDML', zorder=0,
    linestyle='-', color='#00946F', linewidth=2
)
ax.plot(
    r_exp, g_exp, label=exp_label, zorder=1,
    linestyle=(0, (5, 4)), color='#93F2D4', linewidth=2
)
ax.axhline(1.0, zorder=-1, alpha=1.0, color='silver', linestyle=(0, (1, 4)))

ax.set_xlabel('$r$ (A)')
ax.set_xlim(0, r_max)

ax.set_ylabel('g$_{C_NC_N}$($r$)')
ax.set_ylim(ymin=0)

plt.legend(frameon=False)
plt.savefig(save_path, dpi=600)
