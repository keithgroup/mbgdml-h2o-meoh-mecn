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
Sample structures from a data set.
"""

import os

import numpy as np
from mbgdml import criteria
from mbgdml.data import dataSet


##### Setup #####

# Dset directories.
dest_dir = '../../data/datasets'
dset_dir_h2o = f'{dest_dir}/h2o'
dest_dir_mecn = f'{dest_dir}/mecn'
dest_dir_meoh = f'{dest_dir}/meoh'

dset_dir_solvent = dset_dir_h2o  # Selecting the common data set directory.

# Selecting 
dset_sample_path = f'{dset_dir_solvent}/6h2o/6h2o.temelso.etal-dset.npz'

dset_add_name = '6h2o.temelso.etal-dset-mp2.augccpvtz'  # Name (without npz) of the new data set.
save_dir = f'{dset_dir_solvent}/6h2o'  # Directory to save the data set.

size = 6          # The number of entities per structure to sample from the data set.
quantity = 'all'  # The number of structures to sample from the data set.

center_structures = False    # Move each structure so the center of mass is at the origin.
sampling_updates = False     # Print updates about how many structures have been sampled so far.
always_new = False           # If dset exists we always overwrite make a new one.
additional_sampling = True   # Whether we are adding sampling to an existing data set. False if you want a new one.
copy_EF = False              # Whether or not to copy energies and forces to new data set (if sampling from a data set).

# Structural sampling criteria.
criteria = None             # None for nothing, the actual function if it is desired (e.g., criteria.cm_distance_sum)
z_slice = np.array([])      # np.array([]) for nothing.
cutoff = np.array([])   # np.array([]) for nothing.

save = True         # Save the dset as npz.
print_xyz = False  # Create an XYZ file of all dset structures.
profile = False    # Profile the script with cProfile.





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

dset_add_path = f'{save_dir}/{dset_add_name}.npz'

def main():
    dset_sample = dataSet(dset_sample_path)

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(dset_add_path):
        if always_new:
            dset_add = dataSet()
            dset_add.name = dset_add_name
        elif additional_sampling:
            print('Data set exists; will perform additional sampling.')
            dset_add = dataSet(dset_add_path)
            # Make backup of the original data set if it exists.
            print(f'Making backup of {dset_add_name}')
            dset_add.save(dset_add_name + '-backup', dset_add.asdict, save_dir)
        else:
            print('Data set exits and additional_sampling is false.')
            print('Will not sample more structures.')
            exit()
    else:
        dset_add = dataSet()
        dset_add.name = dset_add_name

    # Sampling from dset_sample.
    dset_add.sample_structures(
        dset_sample, quantity, size,
        criteria=criteria, z_slice=z_slice, cutoff=cutoff,
        center_structures=center_structures,
        sampling_updates=sampling_updates, copy_EF=copy_EF
    )

    if save:
        print(f'Saving {dset_add_path}.')
        dset_add.save(dset_add.name, dset_add.asdict, save_dir)

    print(f'Data set MD5: {dset_add.md5}')

    if print_xyz:
        dset_add.write_xyz(save_dir)
    
    # DEBUG
    

if profile:
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats()
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(50)
else:
    main()