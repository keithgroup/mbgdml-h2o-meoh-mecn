# MIT License
# 
# Copyright (c) 2022, Alex M. Maldonado
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ase import Atoms
import numpy as np
import os
from mbgdml.utils import get_files
from sklearn.preprocessing import StandardScaler
from dscribe.descriptors import MBTR
import umap


system_labels = ['2mer', '3mer']
n_neighbors = 15
min_dist = 0.001
random_state = 1478984482

data_dir = 'analysis/feature-space-dim-red'


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
data_dir = os.path.join(base_dir, data_dir)

npz_paths = get_files(data_dir, '.npz')
umap_npz_paths = [path for path in npz_paths if 'umap' in path]


for system_label in system_labels:
    save_path = f'analysis/feature-space-dim-red/16meoh.{system_label}-geometry-embed-mbtr.npy'
    save_path = os.path.join(base_dir, save_path)

    sys_npz_paths = [path for path in umap_npz_paths if system_label in path]
    npz_path = sys_npz_paths[0]

    umap_data = dict(np.load(npz_path, allow_pickle=True))

    mbtr = MBTR(
        species=['H', 'C', 'N'],
        k1={
            'geometry': {'function': 'atomic_number'},
            'grid': {'min': 0, 'max': 10, 'n': 100, 'sigma': 0.1},
        },
        k2={
            'geometry': {'function': 'inverse_distance'},
            'grid': {'min': 0, 'max': 10, 'n': 100, 'sigma': 0.1},
            'weighting': {'function': 'exp', 'scale': 0.5, 'threshold': 1e-3},
        },
        k3={
            'geometry': {'function': 'cosine'},
            'grid': {'min': 0, 'max': 180, 'n': 360, 'sigma': 3},
            'weighting': {'function': 'exp', 'scale': 0.5, 'threshold': 1e-3},
        },
        periodic=False,
        normalization='l2_each',
    )

    Z = umap_data['Z']
    R = umap_data['R']
    E = umap_data['E']
    n_atoms = len(Z)
    del umap_data

    all_atoms = [Atoms(numbers=Z, positions=r) for r in R]

    mbtr_output = mbtr.create(all_atoms, n_jobs=8)
    feat_norm = StandardScaler().fit_transform(mbtr_output)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state,
        metric='manhattan', n_components=1
    )

    embedding = reducer.fit_transform(X=feat_norm).T[0]
    np.save(save_path, embedding)
