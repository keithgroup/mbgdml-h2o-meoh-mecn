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

from ase import Atoms
import json
import numpy as np
import os
from reptar import File
from mbgdml.data import predictSet
from reptar.descriptors import Criteria, com_distance_sum
import schnetpack as spk
import torch
import umap

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
    '16mer'
]
model_path = 'mecn/3mecn/schnet/3mecn.mb-niter5.nfeat128.cut10-train1000.pt'

only_train = True
dset_idx = 0
cutoff = 17.0
train_json_path = 'training-logs/mecn/3mecn.mb/gdml/train1000/training.json'

# Solvent | 2-body | 3-body |
# H2O     |   6    |   10   |
# MeCN    |   9    |   17   |
# MeOH    |   8    |   14   |

use_pset = True
pset_idx = 1
nbody_orders = [3]
e_bad_cutoff = 0.01  # kcal/mol
in_ev = True
pset_path = 'psets/mecn/3mecn/schnet/16mecn.remya.etal.3mecn-pset-48mecn.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz'

include_energies = False
include_gradients = False

n_neighbors = [5, 10, 15, 20, 25, 50]
min_dists = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
random_states = [1181463918, 1698381693, 604112814]  # np.random.randint(2**32 - 1)

sys_name = '16mecn.3mer'
plot_name = f'{sys_name}-schnet-features-umap'
save_dir = f'analysis/feature-space-dim-red/{sys_name}'






###   SCRIPT   ###

hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev


# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../../'
data_dir = os.path.join(base_dir, 'data/')
data_paths = [os.path.join(data_dir, i) for i in data_paths]
train_json_path = os.path.join(base_dir, train_json_path)
model_dir = '../../../../mbgdml-h2o-meoh-mecn-models'
model_path = os.path.join(model_dir, model_path)
pset_path = os.path.join(data_dir, pset_path)
save_dir = os.path.join(base_dir, save_dir)

if only_train:
    with open(train_json_path) as f:
        json_data = json.load(f)
    train_idxs = json_data['training']['idxs']

# Prepare model
device = torch.device('cpu')
model = torch.load(model_path, map_location=device)
spk_converter = spk.data.AtomsConverter(device=device)

# Prepare forward hook
def getInput():
    def hook(model, nn_input, output):
        data_descriptor.append(nn_input[0].detach().numpy().flatten().tolist())
    return hook
h2 = model.output_modules[0].out_net[1].out_net[1].register_forward_hook(getInput())

print('Gathering data')
umap_R = []
umap_E = []
umap_G = []
data_labels = []
data_descriptor = []
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
            desc_kwargs = {
                'entity_ids': rfile.get(f'{group_key}/entity_ids')
            }
            r_criteria = Criteria(com_distance_sum, desc_kwargs, cutoff)
            R_idxs, _ = r_criteria.accept(Z, R)
            R = R[R_idxs][train_idxs]
            E = E[R_idxs][train_idxs]
            G = G[R_idxs][train_idxs]
    
    # E and G are in Eh
    # Convert to kcal/mol
    E *= hartree2kcalmol
    G *= hartree2kcalmol

    for j in range(len(R)):
        data = model(spk_converter(Atoms(positions=R[j], numbers=Z)))

        umap_R.append(R[j].tolist())
        umap_E.append(E[j])
        umap_G.append(G[j].tolist())

        data_labels.append(label)
        if include_energies:
            data_descriptor[-1].append(E[j])
        if include_gradients:
            data_descriptor[-1].extend(G[j].flatten().tolist())

data_labels = np.array(data_labels)
data_descriptor = np.array(data_descriptor)


label_idxs = []
for label in labels:
    label_idxs.append(
        np.argwhere(data_labels == label).T[0]
    )

# We scale all descriptors but the gradients.
scaled_data_descriptor = StandardScaler().fit_transform(data_descriptor)
if include_gradients:
    G_size = len(G[j].flatten().tolist())
    scaled_data_descriptor[:,-G_size:] = data_descriptor[:,-G_size:]

if use_pset:
    pset = predictSet(pset_path)
    E_pred, F_pred = pset.nbody_predictions(nbody_orders)
    E_error = pset.E_true - E_pred  # kcal/mol or eV
    if in_ev:
        E_error_plot = E_error*ev2kcalmol
    else:
        E_error_plot = E_error
    over_e_cutoff = np.argwhere(E_error_plot >= e_bad_cutoff).T[0]
    E_error_bad = E_error_plot[over_e_cutoff]
    print(
        f'{len(over_e_cutoff)} out of {len(E_error_plot)} had over {e_bad_cutoff} kcal/mol error'
    )

save_dir = os.path.join(save_dir, plot_name)
os.makedirs(save_dir, exist_ok=True)

npz_dict = {
    'Z': Z,
    'R': np.array(umap_R),
    'r_unit': 'Angstrom',
    'E': np.array(umap_E),
    'e_unit': 'kcal/mol',
    'G': np.array(umap_G),
    'entity_ids': entity_ids,
    'labels': np.array(data_labels),
    'E_error': E_error_plot,
    'min_dists': np.array(min_dists),
    'n_neighbors': np.array(n_neighbors),
}

for random_state in random_states:
    print(f'Random state: {random_state}')
    npz_dict['random_state'] = np.array(random_state)
    npz_dict['embeddings'] = []
    for n_nbr,min_dist in zip(n_neighbors, min_dists):
        print(f'Embedding with {n_nbr} neighbors')
        reducer = umap.UMAP(
            n_neighbors=n_nbr, min_dist=min_dist, random_state=random_state,
            metric='manhattan'
        )

        embedding = reducer.fit_transform(X=scaled_data_descriptor)
        npz_dict['embeddings'].append(embedding)

    save_path = os.path.join(
        save_dir, str(random_state), plot_name + '.npz'
    )
    save_dir_npz = os.path.join(
        save_dir, f'seed-{random_state}'
    )
    os.makedirs(save_dir_npz, exist_ok=True)
    save_path = os.path.join(
        save_dir_npz, plot_name + '.npz'
    )
    np.savez_compressed(save_path, **npz_dict)
    print()