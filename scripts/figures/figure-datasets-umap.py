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

print('Importing packages')
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap

from mbgdml.data import dataSet
from mbgdml.data import mbModel

from sgdml.utils.desc import Desc

# Model information
model_dir = '../../data/models'
model_dir_h2o = f'{model_dir}/h2o'

# Data set information
dset_dir = '../../data/datasets'
dset_dir_h2o = f'{dset_dir}/h2o'



# CHANGE
# Paths
model_path = f'{model_dir_h2o}/2h2o/112h2o.box.pm.gfn2.md.2h2o-model.mb.train500.npz'
dset_path = f'{dset_dir_h2o}/2h2o/12h2o.su.etal.2h2o-dset.npz'

save_dir = '../../analysis/figures/umap'

# Plot
plot_name = 'umap-112h2o.box.pm.gfn2.md.2h2o.model-12h2o.su.etal.2h2o.dset'
model_color = 'dodgerblue'
dset_color = 'orangered'

# UMAP parameters
n_neighbors = 3  # Default: 15
min_dist = 0.1  # Default: 0.1





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Data
# Model
print('Loading the model')
the_model = mbModel()
the_model.load(model_path)
model_R_desc = the_model.model['R_desc'].T
num_model_R_desc = model_R_desc.shape[0]

# Test data set
print('Calculating test data set descriptors')
dset = dataSet(dset_path)
desc = Desc(dset.n_z)
dset_R_desc, _ = desc.from_R(
    dset.R
)
num_dset_R_desc = dset_R_desc.shape[0]

print('Collecting all descriptors')
all_desc = np.concatenate((model_R_desc, dset_R_desc), axis=0)

# UMAP
reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist
)

print('Calculating UMAP embedding')
embedding = reducer.fit_transform(all_desc)
print(reducer.random_state)

print('Plotting UMAP')
plt.scatter(  # Model
    embedding[:num_model_R_desc, 0],
    embedding[:num_model_R_desc, 1],
    facecolors=model_color,
    edgecolors='none',
    label='Model',
    alpha=0.5
)
plt.scatter(  # data set
    embedding[-num_dset_R_desc:, 0],
    embedding[-num_dset_R_desc:, 1],
    facecolors='none',
    edgecolors=dset_color,
    label='Data Set',
    alpha=0.5
)
plt.gca().set_aspect('equal', 'datalim')
plt.legend()
plt.xticks([])
plt.yticks([])
#plt.savefig(f'{save_dir}/{plot_name}.png', dpi=1000)


