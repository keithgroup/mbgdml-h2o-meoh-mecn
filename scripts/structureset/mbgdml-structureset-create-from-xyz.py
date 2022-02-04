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

import os
import argparse
import numpy as np

from mbgdml.data import structureSet
from mbgdml.utils import get_entity_ids, get_comp_ids

##### Setup #####

xyz_path = '/home/alex/Dropbox/keith/projects/solute-solvent-clusters-dev/clusters/solvent/meoh/20meoh/20meoh.yao.etal/20meoh.yao.etal.all.xyz'
r_unit = 'Angstrom'
solvent = 'meoh'
num_mol = 20

save_dir = '../../data/structuresets/meoh/20meoh/20meoh.yao.etal'
name = '20meoh.yao.etal.all'
overwrite = True

if save_dir[-1] != '/':
    save_dir += '/'

all_same_entities = True

"""
Notes
-----
Assumes structures that have all the same chemical species. Otherwise, you will
have to manually specify ``entity_ids`` and ``comp_ids``.
"""



###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if solvent == 'h2o':
    atoms_per_mol = 3
elif solvent == 'mecn' or solvent == 'meoh':
    atoms_per_mol = 6
else:
    raise ValueError(f'{solvent} is not a valid option.')

if not all_same_entities:
    raise ValueError(f'Only works for structures with all the same chemical species.')

def get_filename(path):
    """The name of the file without the extension from a path.

    If there are periods in the file name, will always remove the last one.

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

def create_structure_set(traj_path, r_unit, entity_ids, comp_ids):
    """

    Parameters
    ----------
    file_path : :obj:`str`
        Path to xyz file.
    r_unit : :obj:`str`
        Units of distance. Options are ``'Angstrom'`` or ``'bohr'`` (defined
        by cclib).
    entity_ids : :obj:`list` [:obj:`int`]
        List of indices starting from ``0`` that specify chemically distinct
        portions of the structure. For example, a water
        dimer would be ``[0, 0, 0, 1, 1, 1]``.
    comp_ids : :obj:`list`
        A nested list with an item for every unique ``entity_id``. Each item
        is a list containing two items. First, the ``entity_id`` as a
        string. Second, a label for the specific chemical species/component.
        For example, two water and one methanol molecules could be
        ``[['0', 'h2o'], ['1', 'h2o'], ['2', 'mecn']]``.
    """
    struct_set = structureSet()
    struct_set.from_xyz(traj_path, r_unit, entity_ids, comp_ids)
    return struct_set



def main():

    traj_name = get_filename(xyz_path)

    # Checks if structure set already exists.
    if os.path.exists(save_dir + traj_name + '.npz') and not overwrite:
        raise FileExistsError('Overwrite is false.')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Creates structure set.
    entity_ids = get_entity_ids(atoms_per_mol, num_mol)
    comp_ids = get_comp_ids(solvent, entity_ids)
    Rset = create_structure_set(xyz_path, r_unit, entity_ids, comp_ids)

    # Saves structure set
    Rset.name = name
    Rset.save(Rset.name, Rset.asdict, save_dir)



main()
