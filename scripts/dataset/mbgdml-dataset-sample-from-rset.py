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
Sample structures from a structure set.
"""

import os

import numpy as np

from mbgdml.data import structureSet
from mbgdml.data import dataSet
from mbgdml.criteria import cm_distance_sum


##### Setup #####

# Rset directory paths
rest_dir = '../../data/structuresets'
rset_dir_h2o = f'{rest_dir}/h2o'
rest_dir_mecn = f'{rest_dir}/mecn'
rest_dir_meoh = f'{rest_dir}/meoh'

# Dset directory paths
dest_dir = '../../data/datasets'
dset_dir_h2o = f'{dest_dir}/h2o'
dest_dir_mecn = f'{dest_dir}/mecn'
dest_dir_meoh = f'{dest_dir}/meoh'

rset_dir_solvent = rest_dir_meoh
dset_dir_solvent = dest_dir_meoh

solvent = 'meoh'
rset_paths = [
    f'{rset_dir_solvent}/6meoh/6meoh.md/6meoh.boyd.etal.1.md.orca.mp2.def2tzvp.300k.npz',
]
dset_name = f'6meoh.boyd.etal.1.md.orca.mp2.def2tzvp.300k-dset'

size = 6
save_dir = f'{dset_dir_solvent}/{size}{solvent}/'

# How many structures to sample from the rset?
# A number (e.g., `5000`) or `'all'`.
quantity = 'all'

# Any structure critera for sampling?
r_criteria = None  # None for 1mer, cm_distance_sum for others.
# z_slice is taken care of in script (not used for cm_distance_sum).
cutoff = np.array([])  # Angstroms; [] for 1mer, [##] for others.

# Will translate the center of mass of the sampled cluster to the origin.
center_structures = False

# Will print sampling updates (slows the script down).
sampling_updates = False

# Will overwrite the data set if it already exists.
overwrite = True

# If the data set already exists and overwrite is `True` will append newly
# sampled structures.
additional_sampling_ok = False

# Save the data set?
save = True

# Prints an xyz file of all sampled structures.
print_xyz = True

# Allows only sampling from a specific structure in the rset.
# :obj:`None` if you want to sample from all structures.
# :obj:`int` to specify the structure index of R to sample from.
# For example, `-1` to sample from only the last structure.
single_structure = None




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if save_dir[-1] != '/':
    save_dir += '/'

os.makedirs(save_dir, exist_ok=True)

dset_path = save_dir + dset_name + '.npz'

molecule_sizes = {
    'h2o': 3,
    'mecn': 6,
    'meoh': 6
}
# The atom closest to the center of mass.
criteria_indices = {
    'h2o': 0,  # O, H, H
    'mecn': 4,  # C, H, H, H, C, N
    'meoh': 0  # O, H, C, H, H, H
}

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
    
def get_z_slice(atom_index, molecule_size, n_molecules):
    """Determines z_slice based on size of the cluster.

    Parameters
    ----------
    atom_index : :obj:`int`
        The index of the atom to be used in the structure criteria.
    molecule_size : :obj:`int`
        The number of atoms in a single molecule.
    n_molecules : :obj:`int`
        Number of molecules.
    
    Returns
    """
    z_slice = [atom_index]
    for _ in range(n_molecules - 1):
        z_slice.append(z_slice[-1] + molecule_size)
    return np.array(z_slice)

# Creates z_slice based on size of the cluster.
z_slice = get_z_slice(criteria_indices[solvent], molecule_sizes[solvent], size)

def main():

    if os.path.exists(dset_path):
        if overwrite and additional_sampling_ok:
            print(f'Will perform additional sampling instead of creating a new data set')
            dset = dataSet(dset_path)
        elif overwrite and not additional_sampling_ok:
            # This means we wanted a new data set, not additional sampling.
            print(f'Will overwrite with a new data set')
            dset = dataSet()
            dset.name = dset_name
        else:
            raise FileExistsError(dset_path)
    else:
        dset = dataSet()
        dset.name = dset_name
    
    print(f'Working on {dset.name} data set')
    for rset_path in rset_paths:
        print(f'Sampling {get_filename(rset_path)} structure set')
        rset = structureSet(rset_path)

        if single_structure is not None:
            rset.R = rset.R[single_structure]
        
        dset.sample_structures(
            rset, quantity, size, criteria=r_criteria, z_slice=z_slice,
            cutoff=cutoff, center_structures=center_structures,
            sampling_updates=sampling_updates
        )
    print(f'Sampled {dset.R.shape[0]} structures')

    if save:
        print(f'Saving {dset_path}.')
        dset.save(dset.name, dset.asdict, save_dir)
    print(f'Data set MD5: {dset.md5}')

    if print_xyz:
        dset.write_xyz(save_dir)
    
main()
