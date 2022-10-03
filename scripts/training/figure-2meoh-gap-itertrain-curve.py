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

"""Plots the force MSE of an active learned GAP model."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mbgdml.utils import get_files

json_paths = {
    200: 'training-logs/meoh/2meoh.mb/gap/itertrain/train200/find_problematic_indices.json',
    300: 'training-logs/meoh/2meoh.mb/gap/itertrain/train300/find_problematic_indices.json',
    400: 'training-logs/meoh/2meoh.mb/gap/itertrain/train400/find_problematic_indices.json',
    500: 'training-logs/meoh/2meoh.mb/gap/itertrain/train500/find_problematic_indices.json',
    600: 'training-logs/meoh/2meoh.mb/gap/itertrain/train600/find_problematic_indices.json',
    700: 'training-logs/meoh/2meoh.mb/gap/itertrain/train700/find_problematic_indices.json',
    800: 'training-logs/meoh/2meoh.mb/gap/itertrain/train800/find_problematic_indices.json',
    900: 'training-logs/meoh/2meoh.mb/gap/itertrain/train900/find_problematic_indices.json',
    1000: 'training-logs/meoh/2meoh.mb/gap/itertrain/train1000/find_problematic_indices.json',
}

save_dir = 'analysis/training/'
plot_name = '2meoh.mb-gap-itertrain-force-mse'
fig_types = ['png']

ref_force_rmse = 0.2812708408394224  # kcal/(mol A)
ref_force_mse = ref_force_rmse**2





# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
save_dir = os.path.join(base_dir, save_dir)
os.makedirs(save_dir, exist_ok=True)


# FIGURE #

# Setting up general figure properties
font = {'family' : 'sans-serif',
        'size'   : 8}
plt.rc('font', **font)

mbgdml_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}
color = mbgdml_colors['meoh']
ref_color = 'silver'

print('Collecting data')
train_set_sizes = []
force_mses = []
for train_size,json_path in json_paths.items():
    json_path = os.path.join(base_dir, json_path)
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
    
    all_cluster_force_rmse = np.array(json_dict['clustering']['losses'])
    force_mses.append(np.mean(all_cluster_force_rmse))
    train_set_sizes.append(train_size)

train_set_sizes = np.array(train_set_sizes)
force_mses = np.array(force_mses)

print('Making plot')
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3), constrained_layout=True)

ax.plot(
    train_set_sizes, force_mses, label=None,
    linewidth=1.5, color=color
)
ax.axhline(
    ref_force_rmse, linestyle=(0, (0.2, 3)), color=ref_color, alpha=0.7,
    dash_capstyle='round',
)

ax.set_xlabel('Training Set Size')

ax.set_ylabel('Force MSE (kcal/(mol A))')

for fig_type in fig_types:
    plt.savefig(
        os.path.join(save_dir, plot_name + f'.{fig_type}'), dpi=1000
    )
    plt.close(fig)

