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
Computes and plots a histogram of cluster sizes of a data set.
"""

import os
import numpy as np
from mbgdml.data import dataSet
from mbgdml.criteria import cm_distance_sum
import matplotlib.pyplot as plt

solvent = 'meoh'
dset_path = '../../data/datasets/meoh/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh-dset.mb-cm14.npz'
save_dir = '../../analysis/l-histograms/'



###   SCRIPT   ###

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))



dset = dict(np.load(dset_path, allow_pickle=True))

solvent_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}

# Adding L information to npz.
R = dset['R']
z = dset['z']
z_slice = dset['z_slice']
entity_ids = dset['entity_ids']
cutoff = dset['cutoff'][0]
L = np.zeros(R.shape[0])

for i in range(len(R)):
    L[i] = cm_distance_sum(z, R[i], z_slice, entity_ids, cutoff=None)[1]

dset['L'] = L


# Plotting histogram of L
bin_num = 50
density = False
cumulative = False

hist, bin_edges, _ = plt.hist(
    dset['L'], bins=bin_num, density=density, cumulative=cumulative,
    color=solvent_colors[solvent]
)
bin_width = bin_edges[1] - bin_edges[0]

# X axis
plt.xlabel('$L$ ($\AA$)')
plt.xlim((bin_edges[0] - bin_width, bin_edges[-1] + bin_width))

# Y axis
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{save_dir}{dset["name"]}-L.histo.png')


