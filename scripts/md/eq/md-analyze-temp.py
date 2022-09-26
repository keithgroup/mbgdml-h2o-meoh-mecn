#!/usr/bin/env python3

# MIT License
# 
# Copyright (c) 2021, Alex M. Maldonado
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

"""Analyze temperature equilibration."""

import matplotlib.pyplot as plt
import numpy as np
import os
from reptar import File

exdir_path = 'data/md/h2o/58h2o-mbgdml-md.exdir'
group_key = '1-nvt'
t_setpoint = 298.15
trim_last = 1000  # Do not calculate starting from these last data points.

save_path = f'analysis/md/temps/58h2o-mbgdml-nvt-init_298.15-ttime_50'  # No file extension

###   SCRIPT   ###
base_dir = '/home/alex/Dropbox/keith/projects/mbgdml-h2o-meoh-mecn'
exdir_path = os.path.join(base_dir, exdir_path)
save_path = os.path.join(base_dir, save_path)

# Get data
rfile = File(exdir_path)
temps = rfile.get(f'{group_key}/temp')

avg_prod = np.zeros(temps[:-trim_last].shape)
for i in range(len(avg_prod)):
    avg_prod[i] = np.sum(temps[i:]) / (len(temps)-i)


fig, ax = plt.subplots(1, 1, constrained_layout=True)

ax.plot(temps, zorder=0, marker='o', markersize=1.5, linestyle='', color='#0061AD', linewidth=1.5, alpha=0.1)
ax.axhline(t_setpoint, zorder=-1, alpha=1.0, color='gray', linestyle=(0, (2, 4)))

ax.set_xlabel('Time (fs)')
ax.set_xlim(xmin=0)

ax.set_ylabel('Temperature (K)')

plt.savefig(save_path + '-temps.png', dpi=600)
plt.close()


fig, ax = plt.subplots(1, 1, constrained_layout=True)

ax.plot(avg_prod, zorder=0, linestyle='-', color='#0061AD', linewidth=1.5)
ax.axhline(t_setpoint, zorder=-1, alpha=1.0, color='silver', linestyle=(0, (1, 4)))

ax.set_xlabel('Start of Production (fs)')
ax.set_xlim(xmin=0)

ax.set_ylabel('Mean Temperature (K)')

plt.savefig(save_path + '-prod-start.png', dpi=600)