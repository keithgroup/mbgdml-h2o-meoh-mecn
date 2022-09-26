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

"""Train GDML models"""

import os
import numpy as np
from mbgdml.data import dataSet
from mbgdml.train import mbGDMLTrain, loss_f_e_weighted_mse

# Setting paths.
# dset_path: Path to dataset to train on.
# log_path: Path to directory where training will occur (logs and model will be stored here).
model_name = '62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb'
dset_path = 'meoh/3meoh/gdml/62meoh.sphere.gfn2.md.500k.prod1.3meoh-dset.mb-cm14.npz'

save_dir = os.path.dirname(os.path.abspath(__file__))

# Model parameters
use_sym = True
use_E = True
use_E_cstr = False
use_cprsn = False
use_torch = False

# Iterative training parameters
n_train_init = 200
n_train_final = 1000
n_valid = 2000
n_test = None
model0_path = None
n_train_step = 100
keep_tasks = True

# Bayesian optimization parameters
initial_grid = [
    2, 25, 50, 100, 200, 300, 400, 500, 700, 900, 1100, 1500, 2000, 2500, 3000,
    4000, 5000, 6000, 7000, 8000, 9000, 10000
]
check_energy_pred = True
sigma_bounds = (min(initial_grid), max(initial_grid))
gp_params = {
    'init_points': 10, 'n_iter': 10, 'acq': 'ucb',
    'alpha': 1e-7, 'kappa': 2.0
}
gp_params_final = {
    'init_points': 10, 'n_iter': 20, 'acq': 'ucb',
    'alpha': 1e-7, 'kappa': 2.0
}

# Loss parameters
loss = loss_f_e_weighted_mse
loss_kwargs = {'rho': 0.01}
require_E_eval = True




###   SCRIPT   ###
parent_dir = '../../../'
dset_dir = f'{parent_dir}/data/ml-dsets'
dset_path = os.path.join(dset_dir, dset_path)

dset = dataSet(dset_path)
loss_kwargs['n_atoms'] = dset.n_z

if model0_path is not None:
    model0 = dict(
        np.load(os.path.join(parent_dir, model0_path), allow_pickle=True)
    )
else:
    model0 = None

train = mbGDMLTrain(
    use_sym=use_sym, use_E=use_E, use_E_cstr=use_E_cstr, use_cprsn=use_cprsn
)
train.iterative_train(
    dset, model_name, n_train_init, n_train_final, n_valid, model0=model0,
    n_train_step=n_train_step, sigma_bounds=sigma_bounds, n_test=n_test,
    save_dir=save_dir, initial_grid=initial_grid, gp_params=gp_params,
    gp_params_final=gp_params_final, overwrite=True, write_json=True,
    write_idxs=True, keep_tasks=keep_tasks,
    loss=loss, loss_kwargs=loss_kwargs, require_E_eval=require_E_eval
)

