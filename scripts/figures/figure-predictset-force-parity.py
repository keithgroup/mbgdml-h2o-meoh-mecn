#%%
import os
from mbgdml.data import predictSet
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Predictset paths.
predictset_dir = '../../data/predictsets'
predictset_dir_h2o = f'{predictset_dir}/h2o'

figures_dir = '../../analysis/figures/force-parity'

# CHANGE
predictset_path = f'{predictset_dir_h2o}/3h2o/12h2o.su.etal.3h2o-pset-112h2o.box.pm.gfn2.md.train500.npz'
save_dir = f'{figures_dir}/h2o/12h2o.su.etal'
mb_orders = [1, 2, 3]



###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(save_dir, exist_ok=True)

def get_filename(path):
    """The name of the file without the extension from a path.

    If there are periods in the file name with no file extension, will always
    remove the last one.

    Parameters
    ----------
    path : :obj:`str`
        Path to file.

    Returns
    -------
    :obj:`str`
        The file name without an extension.
    """
    return os.path.splitext(os.path.basename(path))[0]

def calc_errors(true, predicted):
    true = true.flatten()
    predicted = predicted.flatten()

    error = np.subtract(predicted, true)
    abs_error = np.absolute(error)

    max_error = np.max(abs_error)
    rmse = np.sqrt(mean_squared_error(true, predicted))

    return max_error, rmse


########################
def make_figure(mb_order):
    pset = predictSet()
    pset.load(predictset_path)

    try:
        true_E = getattr(pset, 'E_true')
        true_F = getattr(pset, 'F_true')

        predicted_E, predicted_F = pset.nbody_predictions(mb_order)

        max_E_error, rmse_E = calc_errors(true_E, predicted_E)
        max_F_error, rmse_F = calc_errors(true_F, predicted_F)
    except Exception:
        raise Exception


    ##### Figure
    font = {'family': 'sans-serif',
            'size': 8,
            'style': 'normal',
            'weight': 'black'}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=(4, 3.5), constrained_layout=True)

    ax.plot(
        true_F.flatten(), predicted_F.flatten(),
        linestyle='',
        marker='.', markersize=0.5, markerfacecolor='tab:blue', markeredgecolor='tab:blue'
    )


    ax.set_xlabel('True force [kcal/(mol A)]', **font)
    ax.set_ylabel('Predicted force [kcal/(mol A)]', **font)

    limits = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(limits, limits, color='black', alpha=0.15, zorder=0)

    ax.text(
        0.01, 0.90,
        f'Force RMSE: {rmse_F:.3f}\nForce max error: {max_F_error:.3f}',
        size='8', transform=ax.transAxes
    )
    ax.text(
        0.01, 0.80,
        f'Energy RMSE: {rmse_E:.3f}\nEnergy max error: {max_E_error:.3f}',
        size='8', transform=ax.transAxes
    )




    ax.set_aspect('equal')
    step = 50
    half_step = int(step/2)
    ax.set_xticks(list(range(-200, 200 + step, step)))
    ax.set_xticks(list(range(-200, 200 + half_step, half_step)),minor=True)
    ax.set_xticklabels([str(i) for i in range(-200, 200 + half_step, half_step)], minor=True)
    ax.set_yticks(list(range(-200, 200 + step, step)))
    ax.set_yticks(list(range(-200, 200 + half_step, half_step)), minor=True)
    ax.set_yticklabels([str(i) for i in range(-200, 200 + half_step, half_step)], minor=True)
    ax.set_xlim(limits)
    ax.set_ylim(limits)





    plt.savefig(
        f'{save_dir}/{get_filename(predictset_path)}-{mb_order}body.png',dpi=600
    )

    plt.show()

for mb_order in mb_orders:
    make_figure(mb_order)

# %%
