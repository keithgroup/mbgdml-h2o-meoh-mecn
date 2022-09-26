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
colormap_selection = 'plasma'
edge_color = 'black'
edge_width = 0.0
alpha = 1.0
lighten_factor = 0.3


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

dimred_dir = '../../../'
data_dir = os.path.join(dimred_dir, data_dir)

npz_paths = get_files(data_dir, '.npz')
umap_npz_paths = [path for path in npz_paths if 'umap' in path]

# from https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
def lighten_color(color, amount=0.5):
    """Lightens the given color by multiplying (1-luminosity) by the given
    amount.

    Input can be matplotlib color string, hex string, or RGB tuple.
    
    Examples
    --------
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])

for path in umap_npz_paths:
    print(f'Working on {path}')
    umap_dir = os.path.dirname(path)
    umap_data = dict(np.load(path, allow_pickle=True))

    embeddings = umap_data['embeddings']
    min_dists = umap_data['min_dists']
    n_neighbors = umap_data['n_neighbors']
    E = umap_data['E']
    labels = umap_data['labels']
    del umap_data

    # Get label indices
    label_train = labels[0]
    label_16mer = labels[-1]
    idxs_train = np.argwhere(labels == label_train).T[0]
    idxs_16mer = np.argwhere(labels == label_16mer).T[0]

    # Setup colormap for energies
    E_min, E_max, E_mean = np.min(E), np.max(E), np.mean(E)
    norm = mpl.colors.Normalize(vmin=E_min, vmax=E_max)
    cmap = mpl.cm.get_cmap(colormap_selection)
    colors = cmap(norm(E))

     # Lighten colors
    colors_train = np.array([
        lighten_color(colors[i][:3], amount=lighten_factor) for i in idxs_train
    ])
    colors_16mer = colors[idxs_16mer]

    # Plot the values that are furthest away from mean on top
    order_train = np.argsort(np.abs(E[idxs_train] - E_mean))
    order_16mer = np.argsort(np.abs(E[idxs_16mer] - E_mean))

    for n_nbr,min_dist,embedding in zip(n_neighbors, min_dists, embeddings):

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        
        sc = ax.scatter(
            embedding[:, 0][idxs_train][order_train],
            embedding[:, 1][idxs_train][order_train],
            marker='o', s=50,
            alpha=alpha,
            c=colors_train[order_train],
            linewidths=edge_width,
            edgecolors=edge_color,
        )
        ax.scatter(
            embedding[:, 0][idxs_16mer][order_16mer],
            embedding[:, 1][idxs_16mer][order_16mer],
            marker='^', s=30,
            alpha=alpha,
            c=colors_16mer[order_16mer],
            linewidths=edge_width,
            edgecolors=edge_color,
        )

        plt.gca().set_aspect('equal', 'datalim')
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
            label='$n$-body energy (kcal/mol)', location='right'
        )

        save_path = os.path.join(
            umap_dir, f'energy-nnbr{n_nbr}-mind{min_dist}.png'
        )
        plt.savefig(save_path, dpi=1000)

        plt.close()

