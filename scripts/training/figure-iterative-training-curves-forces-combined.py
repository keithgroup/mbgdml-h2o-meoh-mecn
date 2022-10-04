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

"""Plots force RMSEs of models trained with increasing more data (in one plot)."""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mbgdml.data import predictSet
import json

solvents = ['h2o', 'mecn', 'meoh']  # 'h2o', 'mecn', 'meoh'
save_dir = '../../analysis/training/forces/'
image_types = ['svg', 'eps', 'png']





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(save_dir, exist_ok=True)

# Plot names
plot_names = {
    'h2o': 'h2o-mbgdml-training-curve',
    'mecn': 'mecn-mbgdml-training-curve',
    'meoh': 'meoh-mbgdml-training-curve',
}

# Iterative training paths.
iter_train_dir = '../../training-logs/'

training_jsons = {
    'h2o': [
        (
            os.path.join(iter_train_dir, 'h2o/1h2o/gdml/train200/training.json'),
            os.path.join(iter_train_dir, 'h2o/1h2o/gdml/train300/training.json'),
            os.path.join(iter_train_dir, 'h2o/1h2o/gdml/train400/training.json'),
            os.path.join(iter_train_dir, 'h2o/1h2o/gdml/train500/training.json'),
            os.path.join(iter_train_dir, 'h2o/1h2o/gdml/train600/training.json'),
            os.path.join(iter_train_dir, 'h2o/1h2o/gdml/train700/training.json'),
            os.path.join(iter_train_dir, 'h2o/1h2o/gdml/train800/training.json'),
            os.path.join(iter_train_dir, 'h2o/1h2o/gdml/train900/training.json'),
            os.path.join(iter_train_dir, 'h2o/1h2o/gdml/train1000/training.json'),
        ),
        (
            os.path.join(iter_train_dir, 'h2o/2h2o.mb/gdml/train200/training.json'),
            os.path.join(iter_train_dir, 'h2o/2h2o.mb/gdml/train300/training.json'),
            os.path.join(iter_train_dir, 'h2o/2h2o.mb/gdml/train400/training.json'),
            os.path.join(iter_train_dir, 'h2o/2h2o.mb/gdml/train500/training.json'),
            os.path.join(iter_train_dir, 'h2o/2h2o.mb/gdml/train600/training.json'),
            os.path.join(iter_train_dir, 'h2o/2h2o.mb/gdml/train700/training.json'),
            os.path.join(iter_train_dir, 'h2o/2h2o.mb/gdml/train800/training.json'),
            os.path.join(iter_train_dir, 'h2o/2h2o.mb/gdml/train900/training.json'),
            os.path.join(iter_train_dir, 'h2o/2h2o.mb/gdml/train1000/training.json'),
        ),(
            os.path.join(iter_train_dir, 'h2o/3h2o.mb/gdml/train200/training.json'),
            os.path.join(iter_train_dir, 'h2o/3h2o.mb/gdml/train300/training.json'),
            os.path.join(iter_train_dir, 'h2o/3h2o.mb/gdml/train400/training.json'),
            os.path.join(iter_train_dir, 'h2o/3h2o.mb/gdml/train500/training.json'),
            os.path.join(iter_train_dir, 'h2o/3h2o.mb/gdml/train600/training.json'),
            os.path.join(iter_train_dir, 'h2o/3h2o.mb/gdml/train700/training.json'),
            os.path.join(iter_train_dir, 'h2o/3h2o.mb/gdml/train800/training.json'),
            os.path.join(iter_train_dir, 'h2o/3h2o.mb/gdml/train900/training.json'),
            os.path.join(iter_train_dir, 'h2o/3h2o.mb/gdml/train1000/training.json'),
        ),
    ],
    'mecn': [
        (
            os.path.join(iter_train_dir, 'mecn/1mecn/gdml/train200/training.json'),
            os.path.join(iter_train_dir, 'mecn/1mecn/gdml/train300/training.json'),
            os.path.join(iter_train_dir, 'mecn/1mecn/gdml/train400/training.json'),
            os.path.join(iter_train_dir, 'mecn/1mecn/gdml/train500/training.json'),
            os.path.join(iter_train_dir, 'mecn/1mecn/gdml/train600/training.json'),
            os.path.join(iter_train_dir, 'mecn/1mecn/gdml/train700/training.json'),
            os.path.join(iter_train_dir, 'mecn/1mecn/gdml/train800/training.json'),
            os.path.join(iter_train_dir, 'mecn/1mecn/gdml/train900/training.json'),
            os.path.join(iter_train_dir, 'mecn/1mecn/gdml/train1000/training.json'),
        ),
        (
            os.path.join(iter_train_dir, 'mecn/2mecn.mb/gdml/train200/training.json'),
            os.path.join(iter_train_dir, 'mecn/2mecn.mb/gdml/train300/training.json'),
            os.path.join(iter_train_dir, 'mecn/2mecn.mb/gdml/train400/training.json'),
            os.path.join(iter_train_dir, 'mecn/2mecn.mb/gdml/train500/training.json'),
            os.path.join(iter_train_dir, 'mecn/2mecn.mb/gdml/train600/training.json'),
            os.path.join(iter_train_dir, 'mecn/2mecn.mb/gdml/train700/training.json'),
            os.path.join(iter_train_dir, 'mecn/2mecn.mb/gdml/train800/training.json'),
            os.path.join(iter_train_dir, 'mecn/2mecn.mb/gdml/train900/training.json'),
            os.path.join(iter_train_dir, 'mecn/2mecn.mb/gdml/train1000/training.json'),
        ),
        (
            os.path.join(iter_train_dir, 'mecn/3mecn.mb/gdml/train200/training.json'),
            os.path.join(iter_train_dir, 'mecn/3mecn.mb/gdml/train300/training.json'),
            os.path.join(iter_train_dir, 'mecn/3mecn.mb/gdml/train400/training.json'),
            os.path.join(iter_train_dir, 'mecn/3mecn.mb/gdml/train500/training.json'),
            os.path.join(iter_train_dir, 'mecn/3mecn.mb/gdml/train600/training.json'),
            os.path.join(iter_train_dir, 'mecn/3mecn.mb/gdml/train700/training.json'),
            os.path.join(iter_train_dir, 'mecn/3mecn.mb/gdml/train800/training.json'),
            os.path.join(iter_train_dir, 'mecn/3mecn.mb/gdml/train900/training.json'),
            os.path.join(iter_train_dir, 'mecn/3mecn.mb/gdml/train1000/training.json'),
        ),
    ],
    'meoh': [
        (
            os.path.join(iter_train_dir, 'meoh/1meoh/gdml/train200/training.json'),
            os.path.join(iter_train_dir, 'meoh/1meoh/gdml/train300/training.json'),
            os.path.join(iter_train_dir, 'meoh/1meoh/gdml/train400/training.json'),
            os.path.join(iter_train_dir, 'meoh/1meoh/gdml/train500/training.json'),
            os.path.join(iter_train_dir, 'meoh/1meoh/gdml/train600/training.json'),
            os.path.join(iter_train_dir, 'meoh/1meoh/gdml/train700/training.json'),
            os.path.join(iter_train_dir, 'meoh/1meoh/gdml/train800/training.json'),
            os.path.join(iter_train_dir, 'meoh/1meoh/gdml/train900/training.json'),
            os.path.join(iter_train_dir, 'meoh/1meoh/gdml/train1000/training.json'),
        ),
        (
            os.path.join(iter_train_dir, 'meoh/2meoh.mb/gdml/train200/training.json'),
            os.path.join(iter_train_dir, 'meoh/2meoh.mb/gdml/train300/training.json'),
            os.path.join(iter_train_dir, 'meoh/2meoh.mb/gdml/train400/training.json'),
            os.path.join(iter_train_dir, 'meoh/2meoh.mb/gdml/train500/training.json'),
            os.path.join(iter_train_dir, 'meoh/2meoh.mb/gdml/train600/training.json'),
            os.path.join(iter_train_dir, 'meoh/2meoh.mb/gdml/train700/training.json'),
            os.path.join(iter_train_dir, 'meoh/2meoh.mb/gdml/train800/training.json'),
            os.path.join(iter_train_dir, 'meoh/2meoh.mb/gdml/train900/training.json'),
            os.path.join(iter_train_dir, 'meoh/2meoh.mb/gdml/train1000/training.json'),
        ),
        (
            os.path.join(iter_train_dir, 'meoh/3meoh.mb/gdml/train200/training.json'),
            os.path.join(iter_train_dir, 'meoh/3meoh.mb/gdml/train300/training.json'),
            os.path.join(iter_train_dir, 'meoh/3meoh.mb/gdml/train400/training.json'),
            os.path.join(iter_train_dir, 'meoh/3meoh.mb/gdml/train500/training.json'),
            os.path.join(iter_train_dir, 'meoh/3meoh.mb/gdml/train600/training.json'),
            os.path.join(iter_train_dir, 'meoh/3meoh.mb/gdml/train700/training.json'),
            os.path.join(iter_train_dir, 'meoh/3meoh.mb/gdml/train800/training.json'),
            os.path.join(iter_train_dir, 'meoh/3meoh.mb/gdml/train900/training.json'),
            os.path.join(iter_train_dir, 'meoh/3meoh.mb/gdml/train1000/training.json'),
        ),
    ]
}

training_sizes = np.array([i*100 for i in range(2, 11)])

if save_dir[-1] != '/':
    save_dir += '/'

mbgdml_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}
ref_color = 'silver'

# More information: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
use_rc_params = True
font_dirs = ['../../fonts/roboto']
rc_json_path = '../matplotlib-rc-params.json'



##################
###   FIGURE   ###
##################

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

include_ref_values = False

line_styles = ['-', 'dashed', (0, (5, 10))]
markers = ['o', 's', '^']

fig, axes = plt.subplots(1, len(solvents), figsize=(6.5, 3.2), constrained_layout=True)

i = 1
for solvent,ax in zip(solvents, axes):

    color = mbgdml_colors[solvent]

    solvent_training_jsons = training_jsons[solvent]
    
    for i_nbody in range(len(solvent_training_jsons)):
        all_nbody_jsons = solvent_training_jsons[i_nbody]
        test_loss = np.zeros(len(all_nbody_jsons))
        for i_training_size in range(len(all_nbody_jsons)):
            with open(all_nbody_jsons[i_training_size], 'r') as f:
                json_data = json.load(f)
            loss = json_data['testing']['force']['rmse']
            test_loss[i_training_size] = loss  # kcal/mol
        

        ax.plot(
            training_sizes, test_loss,
            marker=markers[i_nbody], markersize=5,
            linestyle=line_styles[i_nbody], linewidth=1.5,
            color=color, label=f'{i_nbody+1}-body'
        )
    
    panel_label = chr(ord('@')+i)
    ax.text(0.1, 0.94, panel_label, fontweight='bold', fontsize='x-large', transform=ax.transAxes)
    ax.set_xticks([200, 400, 600, 800, 1000])

    major_step = 0.1
    minor_step = 0.05
    _, y_lim_top = ax.get_ylim()

    ax.set_yticks(
        np.arange(start=0.0, stop=y_lim_top+major_step*2, step=major_step)
    )
    ax.set_yticks(
        np.arange(start=minor_step, stop=y_lim_top+major_step*2, step=minor_step),
        minor=True
    )
    ax.set_ylim(bottom=0.0, top=y_lim_top)
    ax.legend(frameon=False)
    
    i += 1


axes[1].set_xlabel('Training Set Size')
axes[0].set_ylabel('Force RMSE (kcal/(mol $\AA$))')


print(f'Saving {plot_names[solvent]}')
for image_type in image_types:
    plt_path = os.path.join(save_dir, f'all-mbgdml-training-curves-forces.{image_type}')
    plt.savefig(plt_path, dpi=1000)