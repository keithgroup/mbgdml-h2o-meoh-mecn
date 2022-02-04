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

"""Determines origin of energy errors (cutoff or prediction)."""

import numpy as np
from numpy.lib.function_base import average
import scipy.optimize as optimize
from mbgdml.data import predictSet


# Predict set dirs.
predictset_dir = '../../data/predictsets'
predictset_dir_h2o = f'{predictset_dir}/h2o'
predictset_dir_mecn = f'{predictset_dir}/mecn'
predictset_dir_meoh = f'{predictset_dir}/meoh'

predictset_dir_solvent = predictset_dir_meoh


mbgdml_pset_paths = [
    f'{predictset_dir_solvent}/1meoh/4meoh.boyd.etal.dset.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.iterativetrain1000.npz',
    f'{predictset_dir_solvent}/2meoh/4meoh.boyd.etal.dset.2meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.iterativetrain1000.npz',
    f'{predictset_dir_solvent}/3meoh/4meoh.boyd.etal.dset.3meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.iterativetrain1000.npz',
]


number_structures = 12

# Reparameterize the force integration constant for energy
repara_E_int_constant = False
optimize_with = 'rmse'  # rmse or mae




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

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


def main():
    mbgdml_psets = [predictSet(i) for i in mbgdml_pset_paths]

    # Variables for prediction of all.

    cutoff_E_error = []
    prediction_E_error = []
    prediction_E_error_unsigned = []  # Total absolute error
    total_E_error = []
    repara_pred_E_error = []
    repara_total_E_error = []

    for pset in mbgdml_psets:
        nbody_order = pset.models_order
        assert len(nbody_order) == 1
        nbody_order = nbody_order[0]

        E_nbody, _ = pset.nbody_predictions([nbody_order])
        E_nbody = E_nbody

        E_true = pset.E_true

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

            if repara_E_int_constant:
                pred_mae_nbody = compute_E_pred_mae(0.0, E_true, E_nbody)
                pred_rmse_nbody = compute_E_pred_rmse(0.0, E_true, E_nbody)

                if optimize_with == 'mae':
                    opt_function = compute_E_pred_mae
                elif optimize_with == 'rmse':
                    opt_function = compute_E_pred_rmse
                res = optimize.minimize(
                    opt_function,
                    x0=0.0,
                    args=(E_true, E_nbody),
                )
                int_constant_change_opt = res.x[0]
                pred_mae_nbody_opt = compute_E_pred_mae(int_constant_change_opt, E_true, E_nbody)
                pred_rmse_nbody_opt = compute_E_pred_rmse(int_constant_change_opt, E_true, E_nbody)
                
                print('Optimization')
                print(f'MAE: {pred_mae_nbody:.5f} -> {pred_mae_nbody_opt:.5f} kcal/mol')
                print(f'RMSE: {pred_rmse_nbody:.5f} -> {pred_rmse_nbody_opt:.5f} kcal/mol')

                prediction_E_error_nbody_opt = compute_E_pred_error_sum(int_constant_change_opt, E_true, E_nbody)
                repara_pred_E_error.append(prediction_E_error_nbody_opt)

                total_E_error_nbody_opt = cutoff_E_error_nbody + prediction_E_error_nbody_opt
                repara_total_E_error.append(total_E_error_nbody_opt)

                print(f'c adjustment: {res.x[0]:.10f}')
                print(f'    => Pred. E error: {prediction_E_error_nbody_opt:.3f} kcal/mol ({(prediction_E_error_nbody_opt - prediction_E_error_nbody):.3f} change)')
                print(f'    => Total E error: {total_E_error_nbody_opt:.3f} kcal/mol ({(total_E_error_nbody_opt - total_E_error_nbody):.3f} change)\n')

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
            

    if repara_E_int_constant:
        print(f'\nIf optimize with new integration constants:')
        print(f'    New MBE energy error from predictions: {np.sum(repara_pred_E_error):.3f} kcal/mol ({(np.sum(repara_pred_E_error) - np.sum(prediction_E_error)):.3f} change)')
        print(f'    New MBE energy error: {np.sum(repara_total_E_error):.3f} kcal/mol ({(np.sum(repara_total_E_error) - np.sum(total_E_error)):.3f} change)')
        


    


main()
