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

"""Creates a prediction set"""

import copy
import json
import numpy as np
import os
from reptar import File
from mbgdml.analysis.clustering import get_clustered_data, get_cluster_losses, cluster_loss_F_mse
from mbgdml.analysis.problematic import prob_structures
from mbgdml.data import dataSet
from mbgdml.mbe import mbePredict
from mbgdml.predict import predict_schnet, schnetModel

model_path = 'h2o/3h2o/schnet/3h2o.mb-niter5.nfeat128.cut10-train1000.pt'
clustering_file_path = 'h2o/3h2o.mb/gdml/train1000/find_problematic_indices.json'
dset_path = 'h2o/3h2o/gdml/140h2o.sphere.gfn2.md.500k.prod1.3h2o-dset.mb-cm10.npz'
save_path = 'h2o/3h2o.mb/schnet/'

comp_ids = ['h2o', 'h2o', 'h2o']
device = 'cpu'
in_ev = True


base_dir = '../../../'
dset_dir = os.path.join(base_dir, 'data/ml-dsets/')
log_dir = os.path.join(base_dir, 'logs-training/')
model_dir = '../../../../mbgdml-h2o-meoh-mecn-models'
save_dir = log_dir

model_path = os.path.join(model_dir, model_path)
dset_path = os.path.join(dset_dir, dset_path)
clustering_file_path = os.path.join(log_dir, clustering_file_path)
save_path = os.path.join(save_dir, save_path)

hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

dset = dataSet(dset_path)
z = dset.z
R = dset.R
entity_ids = dset.entity_ids
comp_ids = dset.comp_ids
E_true = dset.E
F_true = dset.F

with open(clustering_file_path, 'r') as f:
    clustering_info = json.load(f)
cl_indices = clustering_info['clustering']['indices']
cl_pop = clustering_info['clustering']['population']
cl_losses_mbgdml = clustering_info['clustering']['losses']

schnet_model = schnetModel(model_path, comp_ids, device)

mbe_pred = mbePredict([schnet_model], predict_schnet)

E_pred, F_pred = mbe_pred.predict(z, R, entity_ids, comp_ids)
if in_ev:
    E_pred *= ev2kcalmol
    F_pred *= ev2kcalmol

E_errors = E_pred - E_true
F_errors = F_pred - F_true

F_errors_cl = get_clustered_data(cl_indices, F_errors)

loss_kwargs = {'F_errors': F_errors_cl}
cl_losses = get_cluster_losses(
    cluster_loss_F_mse, loss_kwargs
)

clustering_info_new = copy.copy(clustering_info)
clustering_info_new['clustering']['losses'] = cl_losses
del clustering_info_new['problematic_clustering']
print(cl_losses-cl_losses_mbgdml)
