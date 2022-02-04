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
Writes slum calculations for missing energies and forces in data sets.
"""

import os
import numpy as np
from mbgdml.data import dataSet
from mbgdml.qc import slurm_engrad_calculation


###   Setup   ###

# Dset paths
dest_dir = '../../data/datasets'
dset_dir_h2o = f'{dest_dir}/h2o'
dest_dir_mecn = f'{dest_dir}/mecn'
dest_dir_meoh = f'{dest_dir}/meoh'

# Calc save dir
calc_save_dir = '../../../mbgdml-solvents-engrads/data-engrads'
save_dir_h2o = f'{calc_save_dir}/h2o'
save_dir_mecn = f'{calc_save_dir}/mecn'
save_dir_meoh = f'{calc_save_dir}/meoh'

# Setup parameters
dset_dir_solvent = dset_dir_h2o
solvent_save_dir = save_dir_h2o

dset_path = f'{dset_dir_solvent}/4h2o/4h2o.temelso.etal-dset-mp2.augccpvtz.npz'  # The data set to make engrad jobs from.
structure_label = '4h2o.temelso.etal'  # Structure label for the engrad calculations.
calc_name = f'{structure_label}-orca.engrad-mp2.augccpvtz'  # Job name.
cluster_name = '4h2o'  # Parent cluster label. Just used to write all engrads in the same directory.
max_calcs = 50  # Maximum number of consecutive calculations to put in one job.


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

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

def prepare_calc(calc_name, z, R, save_dir):
    slurm_engrad_calculation(
        'orca',
        z,
        R,
        calc_name,
        calc_name,
        calc_name,
        theory='MP2',  # MP2; RI-MP2
        basis_set='aug-cc-pVTZ',
        charge=0,
        multiplicity=1,
        cluster='smp',
        nodes=1,
        cores=24,
        days=3,
        hours=00,
        calc_dir=save_dir,
        options='TightSCF FrozenCore',  # TightSCF FrozenCore; def2/J def2-TZVP/C TightSCF FrozenCore
        control_blocks=(
            '%maxcore 8000\n\n'
            '%scf\n    ConvForced true\nend'
        ),
        write=True,
        submit=False
    )

def main():

    dset = dataSet(dset_path)

    missing_engrad_indices = np.argwhere(np.isnan(dset.E))[:,0]

    # Splits up calculations to a maximum of 500 engrads per calculation.
    if len(missing_engrad_indices) > max_calcs:
        start = 0
        end = max_calcs
        while start < len(missing_engrad_indices):
            if end > len(missing_engrad_indices):
                end = len(missing_engrad_indices)
            calc_name_iter = f'{calc_name}-{start}.to.{end-1}'
            save_dir = f'{solvent_save_dir}/{cluster_name}/{structure_label}/{calc_name_iter}'

            if save_dir[-1] != '/':
                save_dir += '/'
            
            os.makedirs(save_dir, exist_ok=True)

            prepare_calc(calc_name_iter, dset.z, dset.R[start:end], save_dir)

            start += max_calcs
            end += max_calcs
        
    else:
        save_dir = f'{solvent_save_dir}/{cluster_name}/{structure_label}/{calc_name}'
        if save_dir[-1] != '/':
            save_dir += '/'
        os.makedirs(save_dir, exist_ok=True)

        prepare_calc(calc_name, dset.z, dset.R[missing_engrad_indices], save_dir)


main()
