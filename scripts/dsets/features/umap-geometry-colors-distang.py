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

import numpy as np
import os
from mbgdml.utils import get_files
from sklearn.preprocessing import StandardScaler
from reptar.descriptors import get_center_of_mass
import itertools


system_labels = ['2mer', '3mer']

data_dir = 'analysis/feature-space-dim-red'


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
data_dir = os.path.join(base_dir, data_dir)

npz_paths = get_files(data_dir, '.npz')
umap_npz_paths = [path for path in npz_paths if 'umap' in path]

descriptor_length = 3

def get_methyl_to_N(r):
    return r[-1]-r[0]  # Z = [6 1 1 1 6 7]

def get_angle(v1, v2):
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_unit, v2_unit)
    angle = np.arccos(dot_product)
    return angle

def _ratio(com_distance, angle):
    return (1.0*angle)/(1.0*com_distance)

def _dist_angs(Z, r, e, g, entity_ids):
    """Distances and angels"""
    z_split = np.split(Z, len(set(entity_ids)), axis=0)
    r_split = np.split(r, len(set(entity_ids)), axis=0)

    coms = [get_center_of_mass(z_split[0], r_i) for r_i in r_split]

    com_distances, angles = [], []

    for i,j in itertools.combinations(range(0, len(r_split)), 2):
        com_distances.append(np.linalg.norm(coms[i] - coms[j]))
        angles.append(
            get_angle(get_methyl_to_N(r_split[i]), get_methyl_to_N(r_split[j]))
        )
    return np.array(com_distances), np.array(angles)

def r_descriptor(dists, angles):
    """Custom descriptor"""
    descriptor = _ratio(dists, angles)
    descriptor = np.sum(descriptor)
    return descriptor
    


for system_label in system_labels:
    save_path = f'analysis/feature-space-dim-red/16meoh.{system_label}-geometry-embed-distang.npy'
    save_path = os.path.join(base_dir, save_path)

    sys_npz_paths = [path for path in umap_npz_paths if system_label in path]
    npz_path = sys_npz_paths[0]

    umap_data = dict(np.load(npz_path, allow_pickle=True))

    Z = umap_data['Z']
    R = umap_data['R']
    E = umap_data['E']
    G = umap_data['G']
    entity_ids = umap_data['entity_ids']
    n_atoms = len(Z)
    del umap_data

    feat = np.empty(len(R))
    #dists = np.empty((len(R), int(len(set(entity_ids)))))
    #angles = np.empty(dists.shape)
    for i in range(len(R)):
        dists, angles = _dist_angs(Z, R[i], E[i], G[i], entity_ids)
        feat[i] = r_descriptor(dists, angles)
    
    np.save(save_path, feat)
