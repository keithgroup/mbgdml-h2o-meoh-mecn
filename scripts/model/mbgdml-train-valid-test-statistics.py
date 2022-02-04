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
Print training, validation, and test set statistics for a mbGDML model.
"""

import numpy as np
from mbgdml.data import dataSet
from mbgdml.predict import mbPredict

data_dir = '../../data'
dset_path = f'{data_dir}/datasets/mecn/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn-dset.mb-cm9.npz'
model_path = f'{data_dir}/models/mecn/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain1000.npz'




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))


dset = dataSet(dset_path)
F_true = dset.F
E_true = dset.E
R = dset.R

model_dict = dict(np.load(model_path, allow_pickle=True))
idxs_train = model_dict['idxs_train']
idxs_valid = model_dict['idxs_valid']
predict = mbPredict([model_path])

def mae(errors):
    return np.mean(np.abs(errors))

def rmse(errors):
    return np.sqrt(np.mean((errors)**2))

# Training 
R_train = R[idxs_train]
F_train_true = F_true[idxs_train]
E_train_true = E_true[idxs_train]
E_train, F_train = predict.predict(dset.z, R_train, dset.entity_ids, dset.comp_ids)

F_train_errors = F_train.flatten() - F_train_true.flatten()
E_train_errors = E_train.flatten() - E_train_true.flatten()

F_train_mae = mae(F_train_errors)
F_train_rmse = rmse(F_train_errors)
E_train_mae = mae(E_train_errors)
E_train_rmse = rmse(E_train_errors)

print(f'Training ({len(E_train)} points)   Forces MAE: {F_train_mae:.4f}  RMSE: {F_train_rmse:.4f}')
print(f'                         Energy MAE: {E_train_mae:.4f}  RMSE: {E_train_rmse:.4f}\n')

# Validation 
R_valid = R[idxs_valid]
F_valid_true = F_true[idxs_valid]
E_valid_true = E_true[idxs_valid]
E_valid, F_valid = predict.predict(dset.z, R_valid, dset.entity_ids, dset.comp_ids)

F_valid_errors = F_valid.flatten() - F_valid_true.flatten()
E_valid_errors = E_valid.flatten() - E_valid_true.flatten()

F_valid_mae = mae(F_valid_errors)
F_valid_rmse = rmse(F_valid_errors)
E_valid_mae = mae(E_valid_errors)
E_valid_rmse = rmse(E_valid_errors)

print(f'Validation ({len(E_valid)} points)    Forces MAE: {F_valid_mae:.4f}  RMSE: {F_valid_rmse:.4f}')
print(f'                            Energy MAE: {E_valid_mae:.4f}  RMSE: {E_valid_rmse:.4f}\n')

# Testing
idx_mask = np.ones(len(E_true), dtype=bool)
idx_mask[idxs_train] = False
idx_mask[idxs_valid] = False

R_test = R[idx_mask]
F_test_true = F_true[idx_mask]
E_test_true = E_true[idx_mask]

E_test, F_test = predict.predict(dset.z, R_test, dset.entity_ids, dset.comp_ids)

F_test_errors = F_test.flatten() - F_test_true.flatten()
E_test_errors = E_test.flatten() - E_test_true.flatten()

F_test_mae = mae(F_test_errors)
F_test_rmse = rmse(F_test_errors)
E_test_mae = mae(E_test_errors)
E_test_rmse = rmse(E_test_errors)

print(f'Testing ({len(E_test)} points)       Forces MAE: {F_test_mae:.4f}  RMSE: {F_test_rmse:.4f}')
print(f'                            Energy MAE: {E_test_mae:.4f}  RMSE: {E_test_rmse:.4f}')
