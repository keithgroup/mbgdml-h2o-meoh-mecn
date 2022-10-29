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

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from reptar import File
from mbgdml.models import gdmlModel
from mbgdml.data import predictSet
from mbgdml.analysis.models import gdml_mat52
from reptar.descriptors import criteria, com_distance_sum
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_paths = [
    'md-sampling/48mecn-xtb.md-samples.exdir',
    'isomers/16mecn-remya.etal.exdir'
]
group_keys = [
    '/3mecn',
    '/samples_3mecn'
]
labels = [
    'Train',
    'Test'
]

only_train = True
dset_idx = 0
cutoff = 17.0
train_json_path = 'training-logs/mecn/3mecn.mb/gdml/train1000/training.json'
model_path = 'mecn/3mecn/gdml/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-train1000.npz'

# Solvent | 2-body | 3-body |
# H2O     |   6    |   10   |
# MeCN    |   9    |   17   |
# MeOH    |   8    |   14   |

use_pset = True
pset_idx = 1
nbody_orders = [3]
e_bad_cutoff = 0.01  # kcal/mol
in_ev = False
pset_path = 'psets/mecn/3mecn/gdml/16mecn.remya.etal.3mecn-pset-48mecn.sphere.gfn2.md.500k.prod1-gdml.train1000.npz'

remove_symmetries = True

colors = ['#ffb81c', '#003594']
plot_alphas = [0.3, 0.6]
fillstyles=['full', 'none']
markeredgewidths = [0, 1.1]

n_components = 30

sys_name = '16mecn.3mer'
plot_name = f'{sys_name}-gdml-features-pca'
save_dir = f'analysis/feature-space-dim-red/{sys_name}'




###   SCRIPT   ###
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev


# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
data_dir = os.path.join(base_dir, 'data/')
data_paths = [os.path.join(data_dir, i) for i in data_paths]
train_json_path = os.path.join(data_dir, train_json_path)
model_dir = '../../../../mbgdml-h2o-meoh-mecn-models'
model_path = os.path.join(model_dir, model_path)
pset_path = os.path.join(data_dir, pset_path)
save_dir = os.path.join(base_dir, save_dir)

if only_train:
    with open(train_json_path) as f:
        json_data = json.load(f)
    train_idxs = json_data['training']['idxs']

model = gdmlModel(model_path)

save_dir = os.path.join(save_dir, plot_name)
os.makedirs(save_dir, exist_ok=True)

###   Feature space   ###

data_labels = []
data_features = []
data_outputs = []  # energies and/or gradients
entity_combs = ((0, 1, 2),)
for i in range(len(data_paths)):
    data_path = data_paths[i]
    group_key = group_keys[i]
    label = labels[i]

    rfile = File(data_path)
    Z = rfile.get(f'{group_key}/atomic_numbers')
    R = rfile.get(f'{group_key}/geometry')
    E = rfile.get(f'{group_key}/energy_ele_nbody_mp2.def2tzvp_orca')
    G = rfile.get(f'{group_key}/grads_nbody_mp2.def2tzvp_orca')
    entity_ids = rfile.get(f'{group_key}/entity_ids')

    if only_train:
        if i == dset_idx:
            desc_arg_keys = ('atomic_numbers', 'geometry', 'entity_ids')
            desc_args = (rfile.get(f'{group_key}/{dkey}') for dkey in desc_arg_keys)
            _, R_idxs = criteria(com_distance_sum, desc_args, cutoff)
            R = R[R_idxs][train_idxs]
            E = E[R_idxs][train_idxs]
            G = G[R_idxs][train_idxs]
    
    # E and G are in Eh
    # Convert to kcal/mol
    E *= hartree2kcalmol
    G *= hartree2kcalmol

    for j in range(len(R)):
        kernel_values = gdml_mat52(model, Z, R[j])
        if remove_symmetries:
            n_per_symm = int(kernel_values.shape[0]/model.n_perms)
            kernel_values = kernel_values[:n_per_symm]

        data_labels.append(label)
        data_features.append(kernel_values)

data_labels = np.array(data_labels)
data_features = np.array(data_features)
data_outputs = np.array(data_outputs)

label_idxs = []
for label in labels:
    label_idxs.append(
        np.argwhere(data_labels == label).T[0]
    )

# Scaling features and running PCA
scaled_data_features = StandardScaler().fit_transform(data_features)

pca = PCA(n_components=n_components, svd_solver='full')
pca.fit(scaled_data_features, y=None)
eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

save_path = os.path.join(
    save_dir, plot_name + '-eigenvalues.csv'
)
with open(save_path, 'w') as f:
    print('component,eigenvalue,')
    f.write('component,eigenvalue,\n')
    for i in range(len(eigenvalues)):
        print(f'{i},{eigenvalues[i]:.2f},')
        f.write(f'{i},{eigenvalues[i]},\n')




if use_pset:
    pset = predictSet(pset_path)
    E_pred, F_pred = pset.nbody_predictions(nbody_orders)
    E_error = pset.E_true - E_pred
    if in_ev:
        E_error *= ev2kcalmol
    over_e_cutoff = np.argwhere(E_error >= e_bad_cutoff).T[0]
    E_error_bad = E_error[over_e_cutoff]
    print(f'{len(over_e_cutoff)} out of {len(E_error)} had over {e_bad_cutoff} kcal/mol error')

# Bar plot of variance ratio
fig, ax = plt.subplots(1, 1, constrained_layout=True)

ax.bar(
    x=np.arange(len(eigenvalues)),
    height=eigenvalues
)

ax.set_xlabel('Component')
ax.set_ylabel('Eigenvalue')

# Save plot
save_path = os.path.join(
    save_dir, plot_name + '-eigenvalues.png'
)
plt.savefig(save_path, dpi=600)
