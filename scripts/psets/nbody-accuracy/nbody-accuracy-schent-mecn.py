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

"""Determines origin of n-body energy errors (cutoff or prediction)."""

from mbgdml.data import PredictSet
import numpy as np
from numpy.lib.function_base import average
import os
import scipy.optimize as optimize
from reptar import File


dset_path = 'isomers/16mecn-remya.etal.exdir'
group_key = '/'

pset_paths = [
    'mecn/1mecn/schnet/16mecn.remya.etal.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz',
    'mecn/2mecn/schnet/16mecn.remya.etal.2mecn-pset-48mecn.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz',
    'mecn/3mecn/schnet/16mecn.remya.etal.3mecn-pset-48mecn.sphere.gfn2.md.500k.prod1-schnet.niter5.nfeat128.best.train1000.npz',
]
in_ev = True


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
data_dir = os.path.join(base_dir, 'data/')
pset_dir = os.path.join(data_dir, 'psets/')
dset_path = os.path.join(data_dir, dset_path)
pset_paths = [os.path.join(pset_dir, i) for i in pset_paths]

hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

def compute_E_pred_error_sum(int_constant_change, E_true, E_pred, unsigned=False):
    """
    """
    E_error = []
    nan_idx = np.argwhere(np.isnan(E_pred)).flatten()
    for i in range(len(E_pred)):
        if i not in nan_idx:
            E_pred_i = E_pred[i] + int_constant_change
            if unsigned:
                e_error = abs(E_pred_i - E_true[i])
            else:
                e_error = E_pred_i - E_true[i]
            E_error.append(e_error)
    E_error_sum = np.nansum(E_error)
    return E_error_sum

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


rfile = File(dset_path)
number_structures = rfile.get(f'{group_key}/geometry').shape[0]

psets = [PredictSet(i, Z_key='z') for i in pset_paths]

# Variables for prediction of all.

cutoff_E_error = []
prediction_E_error = []
prediction_E_error_unsigned = []  # Total absolute error
total_E_error = []
repara_pred_E_error = []
repara_total_E_error = []

for pset in psets:
    nbody_orders = pset.nbody_orders
    assert len(nbody_orders) == 1
    nbody_order = nbody_orders[0]

    E_nbody, _ = pset.nbody_predictions([nbody_order])

    E_true = pset.E_true

    if in_ev:
        E_nbody *= ev2kcalmol
        E_true *= ev2kcalmol

    if number_structures == 1:

        idxs_cutoff = np.argwhere(np.isnan(E_nbody)).flatten()
        cutoff_E_error_nbody = -np.sum(E_true[idxs_cutoff])
        cutoff_E_error.append(cutoff_E_error_nbody)

        prediction_E_error_nbody = compute_E_pred_error_sum(0.0, E_true, E_nbody)
        prediction_E_error.append(prediction_E_error_nbody)
        prediction_E_error_nbody_unsigned = compute_E_pred_error_sum(0.0, E_true, E_nbody, unsigned=True)
        prediction_E_error_unsigned.append(prediction_E_error_nbody_unsigned)
        
        total_E_error_nbody = np.sum(cutoff_E_error_nbody + prediction_E_error_nbody)
        total_E_error.append(total_E_error_nbody)

        print(f'{nbody_order}-body')
        print(f'---------------------')
        print(f'\033[1mCutoff\033[0m energy error (signed): {cutoff_E_error_nbody:.3f} kcal/mol')
        print(f'\033[1mPred.\033[0m energy error (signed): {prediction_E_error_nbody:.3f} kcal/mol     (unsigned): {prediction_E_error_nbody_unsigned:.3f} kcal/mol' + '\n')
        print(f'\033[1mTotal\033[0m energy error (signed): {total_E_error_nbody:.3f} kcal/mol' + '\n')

    elif number_structures > 1:
        cutoff_E_error = []
        prediction_E_error = []
        total_E_error = []

        nbody_per_structure = pset.n_R / number_structures

        for i in range(0, number_structures):
            structure_start = int(i*nbody_per_structure)
            structure_end = int((i+1)*nbody_per_structure)
            idxs_cutoff = np.argwhere(np.isnan(E_nbody[structure_start:structure_end])).flatten()
            cutoff_E_error_nbody = -np.sum(E_true[structure_start:structure_end][idxs_cutoff])
            cutoff_E_error.append(cutoff_E_error_nbody)

            prediction_E_error_nbody = compute_E_pred_error_sum(0.0, E_true[structure_start:structure_end], E_nbody[structure_start:structure_end])
            prediction_E_error.append(prediction_E_error_nbody)
            
            total_E_error_nbody = np.sum(cutoff_E_error_nbody + prediction_E_error_nbody)
            total_E_error.append(total_E_error_nbody)
        
        cutoff_mae = average([abs(i) for i in cutoff_E_error])
        prediction_mae = average([abs(i) for i in prediction_E_error])

        cutoff_E_error_output = list(map(lambda i: '%.3f' % i, cutoff_E_error))
        prediction_E_error_output = list(map(lambda i: '%.3f' % i, prediction_E_error))
        print(f'{nbody_order}-body energy error in each structure')
        print(f'Cutoff     (MAE - {cutoff_mae:.3f}): {cutoff_E_error_output}')
        print(f'Prediction (MAE - {prediction_mae:.3f}): {prediction_E_error_output}\n')
        

