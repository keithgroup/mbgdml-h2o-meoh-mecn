
"""Prints statistics about pairwise distances of atomic systems"""

import numpy as np
import os
from reptar import File
from scipy.spatial.distance import pdist


data_path = 'data/isomers/16meoh-pires.deturi.exdir'
R_key = 'samples_3meoh/geometry'

##################
###   SCRIPT   ###
##################

# Setup paths
base_dir = '/home/alex/Dropbox/keith/projects/mbgdml-h2o-meoh-mecn'
data_path = os.path.join(base_dir, data_path)

def get_pd(R):
    """Computes pairwise distances from atomic positions.
    
    Parameters
    ----------
    R : :obj:`numpy.ndarray`, shape: ``(n_samples, n_atoms, 3)``
        Atomic positions.
    
    Returns
    -------
    :obj:`numpy.ndarray`, shape: ``(n_samples, n_atoms*(n_atoms-1)/2)``
        Pairwise distances of atoms in each structure.
    """
    assert R.ndim == 3
    n_samples, n_atoms, _ = R.shape
    n_pd = int(n_atoms*((n_atoms-1)/2))
    R_pd = np.zeros((n_samples, n_pd))

    for i in range(len(R)):
        R_pd[i] = pdist(R[i])

    return R_pd

rfile = File(data_path)
R = rfile.get(R_key)
all_pd = get_pd(R)
min_pd = np.min(all_pd)
mean_pd = np.mean(all_pd)
std_pd = np.std(all_pd)
max_pd = np.max(all_pd)
print(f'Min:  {min_pd:.3f}')
print(f'Mean: {mean_pd:.3f}')
print(f'Stdev: {std_pd:.3f}')
print(f'Max:  {max_pd:.3f}')
