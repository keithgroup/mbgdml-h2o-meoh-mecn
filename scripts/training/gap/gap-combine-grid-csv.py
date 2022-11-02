import csv
from natsort import natsorted
import numpy as np
import os

csv_name = '1h2o.mb-gap-combined.csv'
search_dir = '../../../training-logs/h2o/1h2o/grid'
n_atoms = 3
rho_loss = 0.01

def get_files(path, expression, recursive=True):
    """Returns paths to all files in a given directory that matches a provided
    expression in the file name.
    
    Parameters
    ----------
    path : :obj:`str`
        Specifies the directory to search.
    expression : :obj:`str`
        Expression to be tested against all file names in ``path``.
    recursive : :obj:`bool`, optional
        Recursively find all files in all subdirectories.
    
    Returns
    -------
    :obj:`list` [:obj:`str`]
        All absolute paths to files matching the provided expression.
    """
    if path[-1] != '/':
        path += '/'
    if recursive:
        all_files = []
        for (dirpath, _, filenames) in os.walk(path):
            index = 0
            while index < len(filenames):
                if dirpath[-1] != '/':
                    dirpath += '/'
                filenames[index] = dirpath + filenames[index]
                index += 1
            all_files.extend(filenames)
        files = []
        for f in all_files:
            if expression in f:
                files.append(f)
    else:
        files = []
        for f in os.listdir(path):
            filename = os.path.basename(f)
            if expression in filename:
                files.append(path + f)
    return files

# Custom function to accept different validation results.
def loss_f_e_weighted_mse(results, rho, n_atoms):
    r"""Computes a combined energy and force loss function.

    .. math::

        l = \frac{\rho}{Q} \left\Vert E - \hat{E} \right\Vert^2
        + \frac{1}{n_{atoms} Q} \sum_{i=0}^{n_{atoms}}
        \left\Vert \bf{F}_i - \widehat{\bf{F}}_i \right\Vert^2,
    
    where :math:`\rho` is a trade-off between energy and force errors,
    :math:`Q` is the number of validation structures, :math:`\Vert \ldots \Vert`
    is the norm, and :math:`\widehat{\;}` is the model prediction of the
    property.

    Parameters
    ----------
    results : :obj:`dict`
        Validation results which contain force and energy MAEs and RMSEs.
    rho : :obj:`float`
        Energy and force trade-off. A recommended value would be in the range
        of ``0.01`` to ``0.1``.
    n_atoms : :obj:`int`
        Number of atoms.
    
    Returns
    -------
    :obj:`float`
        Validation loss.
    """
    F_mse = results['force']['rmse']**2
    E_mse = results['energy']['rmse']**2
    return rho*E_mse + (1/n_atoms)*F_mse

csv_files = get_files(search_dir, '.csv')
csv_files = natsorted(csv_files)  # sorts by indices

csv_data = []
for csv_path in csv_files:
    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        headers = next(csv_reader)
        for row in csv_reader:
            csv_data.append(row)

# Add label
headers.insert(0, 'Label')
for i in range(len(csv_data)):
    csv_data[i].insert(0, i)

# Add loss
headers.insert(3, 'Loss')
losses = []
for i in range(len(csv_data)):
    results = {
        'force': {'rmse': float(csv_data[i][8])},
        'energy': {'rmse': float(csv_data[i][4])}
    }
    l = loss_f_e_weighted_mse(results, rho_loss, n_atoms)
    losses.append(l)
    csv_data[i].insert(3, l)
losses = np.array(losses)

os.chdir(os.path.dirname(os.path.realpath(__file__)))
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows([headers])
    csv_writer.writerows(csv_data)

idx_sort = np.argsort(losses)

print(f'Lowest loss label: {idx_sort[0]}')
