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
from mbgdml.descriptors import com_distance_sum
import matplotlib.pyplot as plt

solvent = 'meoh'
save_dir = '../../analysis/l-histograms/'

# Stores information for the plots we are going to make.
plot_data = {
    '2h2o-dset-l-histogram': {
        'dset_path': 'h2o/2h2o/gdml/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o-dset.mb-cm6.npz',
        'color': '#4ABBF3',
    },
    '3h2o-dset-l-histogram': {
        'dset_path': 'h2o/3h2o/gdml/140h2o.sphere.gfn2.md.500k.prod1.3h2o-dset.mb-cm10.npz',
        'color': '#4ABBF3',
    },
    '2mecn-dset-l-histogram': {
        'dset_path': 'mecn/2mecn/gdml/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn-dset.mb-cm9.npz',
        'color': '#61BFA3',
    },
    '3mecn-dset-l-histogram': {
        'dset_path': 'mecn/3mecn/gdml/48mecn.sphere.gfn2.md.500k.prod1.3mecn-dset.mb-cm17.npz',
        'color': '#61BFA3',
    },
    '2meoh-dset-l-histogram': {
        'dset_path': 'meoh/2meoh/gdml/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh-dset.mb-cm8.npz',
        'color': '#FFB5BA',
    },
    '3meoh-dset-l-histogram': {
        'dset_path': 'meoh/3meoh/gdml/62meoh.sphere.gfn2.md.500k.prod1.3meoh-dset.mb-cm14.npz',
        'color': '#FFB5BA',
    },
}

###   SCRIPT   ###

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

os.makedirs(save_dir, exist_ok=True)
dset_dir = '../../data/ml-dsets'

for plot_key, data in plot_data.items():
    print(f'Working on {plot_key}')
    dset = dict(np.load(os.path.join(dset_dir, data['dset_path']), allow_pickle=True))
    color = data['color']

    # Adding L information to npz.
    R = dset['R']
    z = dset['z']
    z_slice = dset['z_slice']
    entity_ids = dset['entity_ids']
    cutoff = dset['cutoff'][0]
    L = np.zeros(R.shape[0])

    for i in range(len(R)):
        L[i] = com_distance_sum(z, R[i], entity_ids)[0]
    # Plotting histogram of L
    bin_num = 100
    density = False
    cumulative = False

    hist, bin_edges, _ = plt.hist(
        L, bins=bin_num, density=density, cumulative=cumulative, color=color
    )
    bin_width = bin_edges[1] - bin_edges[0]

    # X axis
    plt.xlabel('$L$ (Ang.)')
    plt.xlim((bin_edges[0] - bin_width, bin_edges[-1] + bin_width))

    # Y axis
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, plot_key + '.png'), dpi=600
    )
    plt.close()

