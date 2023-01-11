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

"""
Plots energy and force RMSEs with respect to the kernel length scale
hyperparameter in training-logs.
"""

import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mbgdml.utils import get_files

search_dir = 'training-logs'


# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
search_dir = os.path.join(base_dir, search_dir)

training_json_paths = get_files(search_dir, 'training.json')

force_color = 'dimgray'
energy_color = 'silver'

# More information: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
use_rc_params = True
font_dirs = ['../../fonts/roboto']
rc_json_path = '../matplotlib-rc-params.json'






# FIGURE #

# Setup matplotlib style
if use_rc_params:
    with open(rc_json_path, 'r') as f:
        rc_params = json.load(f)
    font_paths = mpl.font_manager.findSystemFonts(
        fontpaths=font_dirs, fontext='ttf'
    )
    for font_path in font_paths:
        mpl.font_manager.fontManager.addfont(font_path)
    for key, params in rc_params.items():
        plt.rc(key, **params)


for json_path in training_json_paths:
    print(f'Working on {json_path}')
    json_dir = os.path.dirname(json_path)
    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    sigmas = np.array(json_dict['validation']['sigmas'])
    sigma_selected = json_dict['model']['sigma']
    F_rmse = np.array(json_dict['validation']['force']['rmse'])
    E_rmse = np.array(json_dict['validation']['energy']['rmse'])

    fig, ax_F = plt.subplots(1, 1, figsize=(3.2, 3.2), constrained_layout=True)
    ax_E = ax_F.twinx()

    sig_sort = np.argsort(sigmas)

    ln1 = ax_F.plot(
        sigmas[sig_sort], F_rmse[sig_sort], label='Forces',
        linewidth=1.5, color=force_color
    )
    ln2 = ax_E.plot(
        sigmas[sig_sort], E_rmse[sig_sort], label='Energy',
        linewidth=1.5, color=energy_color, linestyle='--'
    )

    ax_F.axvline(
        sigma_selected, linestyle=(0, (0.2, 3)), color='silver', alpha=0.7,
        dash_capstyle='round',
    )

    ax_F.set_xlabel('Sigma')

    ax_F.set_ylabel('Force RMSE [kcal/(mol Å)]')
    ax_E.set_ylabel('Energy RMSE (kcal/mol)')

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax_E.legend(lns, labs, loc=0)

    plt.savefig(
        os.path.join(json_dir, 'fe_rmse_vs_sigma.png'), dpi=600
    )
    plt.close(fig)

