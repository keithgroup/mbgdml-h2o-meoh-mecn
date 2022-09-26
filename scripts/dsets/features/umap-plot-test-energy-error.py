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

import numpy as np
import os
from mbgdml.utils import get_files
import matplotlib as mpl
import matplotlib.pyplot as plt


data_dir = 'analysis/feature-space-dim-red'
train_color = 'lightgrey'
colormap_selection = 'viridis'
edge_color = 'black'
edge_width = 0.00
max_errors = {
    '2mer': 0.07,
    '3mer': 0.3,
}
# TODO: can parse max errors from csv files.




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

dimred_dir = '../../../'
data_dir = os.path.join(dimred_dir, data_dir)

npz_paths = get_files(data_dir, '.npz')
umap_npz_paths = [path for path in npz_paths if 'umap' in path]

for path in umap_npz_paths:
    print(f'Working on {path}')
    umap_dir = os.path.dirname(path)
    umap_data = dict(np.load(path, allow_pickle=True))
    if '2mer' in path:
        max_error = max_errors['2mer']
    if '3mer' in path:
        max_error = max_errors['3mer']

    embeddings = umap_data['embeddings']
    min_dists = umap_data['min_dists']
    n_neighbors = umap_data['n_neighbors']
    E_error = umap_data['E_error']
    labels = umap_data['labels']
    del umap_data

    # Get label indices
    label_train = labels[0]
    label_16mer = labels[-1]
    idxs_train = np.argwhere(labels == label_train).T[0]
    idxs_16mer = np.argwhere(labels == label_16mer).T[0]

    # Setup colormap for energy errors
    E_error_abs = np.abs(E_error)
    E_error_abs = np.nan_to_num(E_error_abs, 0.0)  # Some energies are not predicted
    norm = mpl.colors.Normalize(vmin=0, vmax=max_error)
    cmap = mpl.cm.get_cmap(colormap_selection)
    colors = cmap(norm(E_error_abs))

    # Plot highest errors on top
    order = np.argsort(E_error_abs)

    for n_nbr,min_dist,embedding in zip(n_neighbors, min_dists, embeddings):

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        # Train scatter
        ax.scatter(
            embedding[:, 0][idxs_train], embedding[:, 1][idxs_train],
            marker='o', s=50,
            alpha=1,
            c=train_color,
            label='Train',
            linewidths=edge_width,
            edgecolors=edge_color,
        )
        # 16mer scatter
        ax.scatter(
            embedding[:, 0][idxs_16mer][order], embedding[:, 1][idxs_16mer][order],
            marker='^', s=30,
            alpha=1,
            c=colors[order],
            label='16mer',
            linewidths=edge_width,
            edgecolors=edge_color,
        )
        plt.gca().set_aspect('equal', 'datalim')
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
            label='Energy error (kcal/mol)', location='right'
        )

        save_path = os.path.join(
            umap_dir, f'error-nnbr{n_nbr}-mind{min_dist}.png'
        )
        plt.savefig(save_path, dpi=1000)

        plt.close()

