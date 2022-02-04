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
Reparameterization of integration constant based on data sets.
"""

import os
import numpy as np
import scipy.optimize as optimize
from mbgdml.data import predictSet

# Data set dirs.
dataset_dir = '../../data/datasets'
dataset_dir_h2o = f'{dataset_dir}/h2o'
dataset_dir_mecn = f'{dataset_dir}/mecn'
dataset_dir_meoh = f'{dataset_dir}/meoh'

# Models dirs.
model_dir = '../../data/models'
model_dir_h2o = f'{model_dir}/h2o'
model_dir_mecn = f'{model_dir}/mecn'
model_dir_meoh = f'{model_dir}/meoh'

dataset_dir_solvent = dataset_dir_meoh
model_dir_solvent = model_dir_meoh

dset_paths = [
    f'{dataset_dir_solvent}/2meoh/boyd.etal/4meoh.boyd.etal.dset.2meoh-dset.mb.npz',
    f'{dataset_dir_solvent}/2meoh/boyd.etal/5meoh.boyd.etal.dset.2meoh-dset.mb.npz',
    f'{dataset_dir_solvent}/2meoh/boyd.etal/6meoh.boyd.etal.dset.2meoh-dset.mb.npz',
    f'{dataset_dir_solvent}/2meoh/16meoh.pires.deturi.2meoh-dset.mb.npz',
    #f'{dataset_dir_solvent}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh-dset.mb-cm8.npz',
]
model_path = f'{model_dir_solvent}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain500.npz'

optimize_with = 'rmse'  # rmse or mae
save = False




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def compute_E_pred_errors(int_constant_change, E_true, E_pred):
    """
    """
    E_errors = []
    nan_idx = np.argwhere(np.isnan(E_pred)).flatten()
    for i in range(len(E_pred)):
        if i not in nan_idx:
            E_pred_i = E_pred[i] + int_constant_change
            E_errors.append(E_pred_i - E_true[i])
    return E_errors

def compute_E_pred_mae(int_constant_change, E_true, E_pred):
    """Computes energy mean absolute error of just predictions.

    Ignores any ``NaN`` values in ``E_pred``.

    Parameters
    ----------
    int_constant_change : :obj:`float`
        Force integration constant modification. ``0.0`` would use the model's
        default constant.
    E_true : :obj:`numpy.ndarray`
        Energy array of all structures. No element should be ``NaN``.
    E_pred : :obj:`numpy.ndarray`
        Energy predictions of the structures in the same order as ``E_pred``.
        This array can have ``NaN`` elements to represent structures outside
        the cutoff.
    
    Returns
    -------
    :obj:`float`
        Mean absolute error of all predictions.
    """
    E_error = []
    nan_idx = np.argwhere(np.isnan(E_pred)).flatten()
    for i in [i for i in range(len(E_pred)) if i not in nan_idx]:
        E_pred_i = E_pred[i] + int_constant_change
        E_error.append(E_pred_i - E_true[i])
    return np.mean(np.abs(E_error))

def compute_E_pred_rmse(int_constant_change, E_true, E_pred):
    """Computes energy root mean squared error of just predictions.

    Ignores any ``NaN`` values in ``E_pred``.

    Parameters
    ----------
    int_constant_change : :obj:`float`
        Force integration constant modification. ``0.0`` would use the model's
        default constant.
    E_true : :obj:`numpy.ndarray`
        Energy array of all structures. No element should be ``NaN``.
    E_pred : :obj:`numpy.ndarray`
        Energy predictions of the structures in the same order as ``E_pred``.
        This array can have ``NaN`` elements to represent structures outside
        the cutoff.
    
    Returns
    -------
    :obj:`float`
        Root mean squared error of all predictions.
    """
    E_error = []
    nan_idx = np.argwhere(np.isnan(E_pred)).flatten()
    for i in [i for i in range(len(E_pred)) if i not in nan_idx]:
        E_pred_i = E_pred[i] + int_constant_change
        E_error.append(E_pred_i - E_true[i])
    rmse = np.sqrt(np.mean(np.array(E_error)**2))
    return rmse


def main():
    # Variables for prediction of all.
    E_true_all = []
    E_pred_all = []

    # Current model constant
    model = dict(np.load(model_path, allow_pickle=True))
    model_int_constant = model['c']

    for dset_path in dset_paths:
        print(f'Working on {os.path.split(dset_path)[1]}')
        pset = predictSet()
        pset.load_dataset(dset_path)
        pset.load_models([model_path])
        pset.prepare(pset.z, pset.R, pset.entity_ids, pset.comp_ids)

        E_nbody, _ = pset.nbody_predictions(pset.models_order)

        E_true_all.extend(pset.E_true)
        E_pred_all.extend(E_nbody)
    
    E_true_all = np.array(E_true_all)
    E_pred_all = np.array(E_pred_all)

    mae_current = compute_E_pred_mae(0.0, E_true_all, E_pred_all)
    rmse_current = compute_E_pred_rmse(0.0, E_true_all, E_pred_all)

    print(f'\nOptimizing with {optimize_with}\n')

    print(f'Current integration constant: {model_int_constant}')
    print(f'     MAE: {mae_current:.5f} kcal/mol')
    print(f'     RMSE: {rmse_current:.5f} kcal/mol')

    if optimize_with == 'mae':
        opt_function = compute_E_pred_mae
    elif optimize_with == 'rmse':
        opt_function = compute_E_pred_rmse
    else:
        raise ValueError(f'{optimize_with} is not a valid selection')
    
    res = optimize.minimize(
        opt_function,
        x0=0.0,
        args=(E_true_all, E_pred_all),
    )
    int_constant_change_optimized = res.x[0]
    int_constant_optimized = model_int_constant + int_constant_change_optimized

    mae_optimized = compute_E_pred_mae(
        int_constant_change_optimized, E_true_all, E_pred_all
    )
    rmse_optimized = compute_E_pred_rmse(
        int_constant_change_optimized, E_true_all, E_pred_all
    )

    print(f'Optimized integration constant: {int_constant_optimized}')
    print(f'     MAE: {mae_optimized:.5f} kcal/mol')
    print(f'     RMSE: {rmse_optimized:.5f} kcal/mol')

    if save:
        save_dir, filename = os.path.split(model_path)
        new_filename = filename[:-4] + '-int.const.opt'
        model['c'] = int_constant_optimized
        new_model_path = save_dir + '/' + new_filename + '.npz'
        print(f'Saving {os.path.split(new_model_path)[1]}')
        np.savez_compressed(new_model_path, **model)
    
    # DEBUG
    #E_errors = compute_E_pred_errors(0.0, E_true_all, E_pred_all)
    #E_errors_opt = compute_E_pred_errors(int_constant_change_optimized, E_true_all, E_pred_all)

main()
