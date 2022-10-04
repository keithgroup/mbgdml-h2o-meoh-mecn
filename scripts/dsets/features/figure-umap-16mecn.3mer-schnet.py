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

"""3mecn energy and geometry descriptor plot for SchNet"""

import numpy as np
import os
from mbgdml.utils import get_files
import matplotlib as mpl
import matplotlib.pyplot as plt

geom_color_path = '16meoh.3mer-geometry-embed-distang.npy'
umap_npz_path = '16mecn.3mer/16mecn.3mer-schnet-features-umap/seed-1181463918/16mecn.3mer-schnet-features-umap.npz'
save_dir = '16mecn.3mer'

train_color = 'lightgrey'
colormap_geometry = 'plasma'
colormap_errors = 'viridis'
edge_color = 'black'
edge_width = 0.00  # 0.05
lighten_factor = 0.3
figsize = (6.7, 2.7)  # Width and height

n_neighbors_plot = [5, 20]
fig_types = ['svg', 'eps']

max_errors = {
    '2mer': 0.07,
    '3mer': 0.3,
}

# More information: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
use_rc_params = True
font_dirs = ['../../../fonts/roboto']
rc_json_path = '../../matplotlib-rc-params.json'





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
data_dir = os.path.join(base_dir, 'analysis/feature-space-dim-red/')
geom_color_path = os.path.join(data_dir, geom_color_path)
umap_npz_path = os.path.join(data_dir, umap_npz_path)
save_dir = os.path.join(data_dir, save_dir)

max_error = max_errors['3mer']

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

# Extract data
umap_data = dict(np.load(umap_npz_path, allow_pickle=True))

embeddings = umap_data['embeddings']
min_dists = umap_data['min_dists']
n_neighbors = umap_data['n_neighbors']
labels = umap_data['labels']
E_error = umap_data['E_error']

# Get label indices
label_train = labels[0]
label_16mer = labels[-1]
idxs_train = np.argwhere(labels == label_train).T[0]
idxs_16mer = np.argwhere(labels == label_16mer).T[0]

del umap_data

# Setup colormap for geometries
geom_colors = np.load(geom_color_path, allow_pickle=True)
norm_geom = mpl.colors.Normalize(vmin=min(geom_colors), vmax=max(geom_colors))
cmap_geom = mpl.cm.get_cmap(colormap_geometry)
colors_geom = cmap_geom(norm_geom(geom_colors))

colors_geom_train = [
    lighten_color(colors_geom[i][:3], amount=lighten_factor) for i in idxs_train
]
colors_geom_16mer = colors_geom[idxs_16mer]

# Setup colormap for energies
E_error_abs = np.abs(E_error)
E_error_abs = np.nan_to_num(E_error_abs, 0.0)  # Some energies are not predicted
norm = mpl.colors.Normalize(vmin=0, vmax=max_error)
cmap = mpl.cm.get_cmap(colormap_errors)
colors_errors = cmap(norm(E_error_abs))

# Plot highest errors on top
error_order = np.argsort(E_error_abs)

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





for n_nbr,min_dist,embedding in zip(n_neighbors, min_dists, embeddings):

    if n_nbr not in n_neighbors_plot:
        continue

    fig, axes = plt.subplots(
        1, 2, constrained_layout=True, figsize=figsize, sharey=True
    )
    ax1, ax2 = axes

    # Errors
    ax1.scatter(
        embedding[:, 0][idxs_train], embedding[:, 1][idxs_train],
        marker='o', s=50,
        alpha=1,
        c=train_color,
        label='Train',
        linewidths=edge_width,
        edgecolors=edge_color,
    )
    ax1.scatter(
        embedding[:, 0][idxs_16mer][error_order], embedding[:, 1][idxs_16mer][error_order],
        marker='^', s=30,
        alpha=1,
        c=colors_errors[error_order],
        label='16mer',
        linewidths=edge_width,
        edgecolors=edge_color,
    )

    # Geometry
    ax2.scatter(
        embedding[:, 0][idxs_train], embedding[:, 1][idxs_train],
        marker='o', s=50,
        alpha=1.0,
        c=colors_geom_train,
        linewidths=edge_width,
        edgecolors=edge_color,
    )
    ax2.scatter(
        embedding[:, 0][idxs_16mer], embedding[:, 1][idxs_16mer],
        marker='^', s=30,
        alpha=1.0,
        c=colors_geom_16mer,
        linewidths=edge_width,
        edgecolors=edge_color,
    )

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1,
        label='Energy error (kcal/mol)', location='right', ticks=None
    )
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_geom, cmap=cmap_geom), ax=ax2,
        label='Geometry descriptor', location='right', ticks=None
    )

    # Subplot labels
    ax1.text(
        0.02, 0.93,
        'A',
        fontsize='large',
        fontweight='bold',
        transform=ax1.transAxes
    )
    ax2.text(
        0.02, 0.93,
        'B',
        fontsize='large',
        fontweight='bold',
        transform=ax2.transAxes
    )

    for fig_type in fig_types:
        save_path = os.path.join(
            save_dir, f'16mecn.3mer-schnet-nnbr{n_nbr}-mind{min_dist}.{fig_type}'
        )
        plt.savefig(save_path, dpi=1000)

    plt.close()

