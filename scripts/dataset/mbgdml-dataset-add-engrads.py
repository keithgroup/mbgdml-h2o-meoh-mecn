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
Script that adds energy+gradient data stored in qcjson files to data sets by
matching structure coordinates.

This script was added to the dataSet class.
"""

import os
import json
import numpy as np
from cclib.parser.utils import convertor
from mbgdml.data import dataSet
from mbgdml.utils import get_files, natsort_list, convert_forces, center_structures


##### Setup #####

# Dset paths
dest_dir = '../../data/datasets'
dset_dir_h2o = f'{dest_dir}/h2o'
dest_dir_mecn = f'{dest_dir}/mecn'
dest_dir_meoh = f'{dest_dir}/meoh'

engrad_dir = '../../../mbgdml-solvents-engrads/data-engrads/engrad-calcs-large-basis'
engrad_dir_h2o = f'{engrad_dir}/h2o'
engrad_dir_mecn = f'{engrad_dir}/mecn'
engrad_dir_meoh = f'{engrad_dir}/meoh'




# Setup parameters

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

dset_dir_solvent = dset_dir_h2o
engrad_dir_solvent = engrad_dir_h2o

dset_path = f'{dset_dir_solvent}/6h2o/6h2o.temelso.etal-dset-mp2.augccpvtz.npz'
engrad_dir = f'{engrad_dir_solvent}/6h2o/6h2o.temelso.etal'

theory = 'mp2.augccpvtz.frozencore'
missing_ok = False  # Expecting missing data after including these engrads.
save_dset = True
# If we need to center the engrad coordinates in order to match to the data set.
need_to_center_engrad_R = False

search_str = '.json'

"""
Notes
-----
If the script is taking a long time with a single value, odds are that the
dset_path and engrads are incorrect. You should just cancel and double check
these values.
"""

##### Script #####

dset_energy_units = 'kcal/mol'

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


def main():

    # Gets all engrad output quantum chemistry json files.
    # For more information: https://github.com/keithgroup/qcjson
    engrad_calc_paths = get_files(engrad_dir, search_str)
    print(f'Found {len(engrad_calc_paths)} engrad calculations')
    # Organizing engrad calculations so earlier structures are loaded first.
    # This does multiple things. Reduces runtime as fewer iterations need to 
    # be made. Also, some data sets may involve trajectories that start from
    # the same structure, so multiple structures might be the same.
    engrad_calc_paths = natsort_list(engrad_calc_paths)

    # Checks to make sure engrads were found.
    if len(engrad_calc_paths) == 0:
        print('No engrads were found')
        exit()

    dset = dataSet(dset_path)
    print(f'Loaded the {dset.name} data set')

    missing_engrad_indices = np.argwhere(np.isnan(dset.E))[:,0].tolist()
    if len(missing_engrad_indices) == 0:
        print(f'There are no missing data')
    elif not missing_ok:
        print(f'There are {len(missing_engrad_indices)} energies not stored in the data set')
    

    z_dset = dset.z
    # Loops thorugh engrad calculations and adds energies and forces for each
    # structure.
    for engrad_calc_path in engrad_calc_paths:
        engrad_name = get_filename(engrad_calc_path)
        print(f'Adding engrads from {engrad_name}')
        # Gets energies and gradients from qcjson.
        with open(engrad_calc_path) as f:
            engrad_data = json.load(f)

        # Loops through all structures that are missing engrads.
        for engrad_i in range(len(engrad_data)):
            _dset_i_to_remove = []
            can_remove = False

            if isinstance(engrad_data, list):
                data = engrad_data[engrad_i]
            else:
                data = engrad_data

            engrad_i_r = np.array(
                data['molecule']['geometry']
            )
            if need_to_center_engrad_R:
                engrad_i_r = center_structures(z_dset, engrad_i_r)

            for dset_i in missing_engrad_indices:
                dset_i_r = dset.R[dset_i]

                # If the atomic coordinates match we assume this is the
                # originating engrad structure.
                # ORCA output files will only include six significant figures
                # for Cartesian coordinates. Sometimes the data sets have more
                # significant figures and the tolerances for allclose were too
                # high, so I had to lower them a little.
                # The defaults were: atol=1e-08, rtol=1e-05
                # Now they are atol=5.1e-07, rtol=0
                # Had to raise it to atol=6.8e-07, rtol=0 for some data sets.
                # Because we used natsort for the engrad calculations they
                # should be in order, but we check anyway.
                if np.allclose(engrad_i_r, dset_i_r, atol=6.8e-07, rtol=0):
                    # Get, convert, and add energies and forces to data set.
                    energy = data['properties']['return_energy']
                    energy = convertor(energy, 'hartree', dset_energy_units)
                    dset.E[dset_i] = energy

                    forces = np.negative(
                        np.array(data['return_result'])
                    )
                    forces = convert_forces(
                        forces, 'hartree', 'Angstrom', dset_energy_units, 'Angstrom'
                    )
                    dset.F[dset_i] = forces

                    # Found the correct structure, so we terminate looking for
                    # the dset structure match.
                    can_remove = True
                    break
                
            # Removes all NaN indices from missing_engrad_indices that have
            # already been matched.
            if can_remove:
                missing_engrad_indices.remove(dset_i)
    
    still_missing = len(np.argwhere(np.isnan(dset.E))[:,0].tolist())
    if still_missing > 0:
        print(f'There are still {still_missing} NaN energies')
        if missing_ok:
            print('Will leave the missing data')
        else:
            print('Missing data was not expected')
            print('Will not save the data set')
            raise ValueError
    
    # Finalizes data set and saves.
    dset.e_unit = dset_energy_units
    dset.theory = theory
    print(f'Data set MD5: {dset.md5}')
    if save_dset:
        print(f'Saving {get_filename(dset_path)}')
        save_dir = os.path.dirname(dset_path) + '/'
        dset.save(dset.name, dset.asdict, save_dir)

main()
