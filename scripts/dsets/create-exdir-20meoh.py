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

"""Example of creating an exdir file with reptar for the 20meoh data."""

import numpy as np
import os
from reptar import File
from reptar.parsers import parserORCA
from mbgdml.utils import get_entity_ids, get_comp_ids


exdir_path = 'isomers/20meoh.yao.etal.exdir'
comp_id = 'meoh'
atoms_per_mol = 6
num_mol = 20

out_dir = 'isomers/20meoh.yao.etal-ouputs/'
out_paths = {
    'def2tzvp': [
        'def2tzvp/20meoh.yao.etal.1-orca-rimp2.def2tzvp.out',
        'def2tzvp/20meoh.yao.etal.2-orca-rimp2.def2tzvp.out',
        'def2tzvp/20meoh.yao.etal.3-orca-rimp2.def2tzvp.out',
        'def2tzvp/20meoh.yao.etal.4-orca-rimp2.def2tzvp.out',
        'def2tzvp/20meoh.yao.etal.5-orca-rimp2.def2tzvp.out',
    ],
    'ccpvtz': [
        'ccpvtz/20meoh.yao.etal.1-orca-rimp2.ccpvtz.out',
        'ccpvtz/20meoh.yao.etal.2-orca-rimp2.ccpvtz.out',
        'ccpvtz/20meoh.yao.etal.3-orca-rimp2.ccpvtz.out',
        'ccpvtz/20meoh.yao.etal.4-orca-rimp2.ccpvtz.out',
        'ccpvtz/20meoh.yao.etal.5-orca-rimp2.ccpvtz.out',
    ],
}


###   SCRIPT   ###
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

# Handling paths
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
data_dir = os.path.join(base_dir, 'data')
out_dir = os.path.join(data_dir, out_dir)
exdir_path = os.path.join(data_dir, exdir_path)

entity_ids = get_entity_ids(atoms_per_mol, num_mol)
comp_ids = get_comp_ids(comp_id, num_mol, entity_ids)

# Parses data using reptar
parsed_data = []
for out_path in out_paths['def2tzvp']:
    out_path = os.path.join(out_dir, out_path)
    parsed_info = parserORCA(out_path=out_path).parse()
    parsed_data.append(parsed_info)

# Make custom data because ... 
combined_data_def2tzvp = {}
for data in parsed_data:
    if 'atomic_numbers' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['atomic_numbers'] = data['system_info']['atomic_numbers']
    if 'charge' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['charge'] = data['system_info']['charge']
    if 'mult' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['mult'] = data['system_info']['mult']
    if 'n_ele' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['n_ele'] = data['system_info']['n_ele']
    if 'hf_type' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['hf_type'] = data['runtime_info']['hf_type']
    
    if 'geometry' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['geometry'] = data['system_info']['geometry']
    else:
        combined_data_def2tzvp['geometry'] = np.vstack(
            (combined_data_def2tzvp['geometry'], data['system_info']['geometry'])
        )
    
    if 'ele_frozen' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['ele_frozen'] = data['outputs']['ele_frozen']
    else:
        combined_data_def2tzvp['ele_frozen'] = np.hstack(
            (combined_data_def2tzvp['ele_frozen'], data['outputs']['ele_frozen'])
        )
    
    if 'energy_scf_rimp2_def2tzvp' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['energy_scf_rimp2_def2tzvp'] = data['outputs']['energy_scf']
    else:
        combined_data_def2tzvp['energy_scf_rimp2_def2tzvp'] = np.vstack(
            (combined_data_def2tzvp['energy_scf_rimp2_def2tzvp'], data['outputs']['energy_scf'])
        )
    
    if 'energy_correl_mp2_def2tzvp' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['energy_correl_mp2_def2tzvp'] = data['outputs']['energy_correl_mp2']
    else:
        combined_data_def2tzvp['energy_correl_mp2_def2tzvp'] = np.vstack(
            (combined_data_def2tzvp['energy_correl_mp2_def2tzvp'], data['outputs']['energy_correl_mp2'])
        )
    
    if 'energy_ele_rimp2_def2tzvp' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['energy_ele_rimp2_def2tzvp'] = data['outputs']['energy_ele']
    else:
        combined_data_def2tzvp['energy_ele_rimp2_def2tzvp'] = np.hstack(
            (combined_data_def2tzvp['energy_ele_rimp2_def2tzvp'], data['outputs']['energy_ele'])
        )
    
    if 'dipole_moment_rimp2_def2tzvp' not in combined_data_def2tzvp.keys():
        combined_data_def2tzvp['dipole_moment_rimp2_def2tzvp'] = data['outputs']['dipole_moment']
    else:
        combined_data_def2tzvp['dipole_moment_rimp2_def2tzvp'] = np.hstack(
            (combined_data_def2tzvp['dipole_moment_rimp2_def2tzvp'], data['outputs']['dipole_moment'])
        )

parsed_data = []
for out_path in out_paths['ccpvtz']:
    out_path = os.path.join(out_dir, out_path)
    parsed_info = parserORCA(out_path=out_path).parse()
    parsed_data.append(parsed_info)

combined_data_ccpvtz = {}
for data in parsed_data:
    if 'energy_scf_rimp2_ccpvtz' not in combined_data_ccpvtz.keys():
        combined_data_ccpvtz['energy_scf_rimp2_ccpvtz'] = data['outputs']['energy_scf']
    else:
        combined_data_ccpvtz['energy_scf_rimp2_ccpvtz'] = np.vstack(
            (combined_data_ccpvtz['energy_scf_rimp2_ccpvtz'], data['outputs']['energy_scf'])
        )
    
    if 'energy_correl_mp2_ccpvtz' not in combined_data_ccpvtz.keys():
        combined_data_ccpvtz['energy_correl_mp2_ccpvtz'] = data['outputs']['energy_correl_mp2']
    else:
        combined_data_ccpvtz['energy_correl_mp2_ccpvtz'] = np.vstack(
            (combined_data_ccpvtz['energy_correl_mp2_ccpvtz'], data['outputs']['energy_correl_mp2'])
        )
    
    if 'energy_ele_rimp2_ccpvtz' not in combined_data_ccpvtz.keys():
        combined_data_ccpvtz['energy_ele_rimp2_ccpvtz'] = data['outputs']['energy_ele']
    else:
        combined_data_ccpvtz['energy_ele_rimp2_ccpvtz'] = np.hstack(
            (combined_data_ccpvtz['energy_ele_rimp2_ccpvtz'], data['outputs']['energy_ele'])
        )
    
    if 'dipole_moment_rimp2_ccpvtz' not in combined_data_ccpvtz.keys():
        combined_data_ccpvtz['dipole_moment_rimp2_ccpvtz'] = data['outputs']['dipole_moment']
    else:
        combined_data_ccpvtz['dipole_moment_rimp2_ccpvtz'] = np.hstack(
            (combined_data_ccpvtz['dipole_moment_rimp2_ccpvtz'], data['outputs']['dipole_moment'])
        )

combined_data_def2tzvp['entity_ids'] = entity_ids
combined_data_def2tzvp['comp_ids'] = comp_ids

# Adds base data from def2-TZVP calculations
rfile = File(exdir_path, mode='w', allow_remove=True, from_dict=combined_data_def2tzvp)

# Adds in the cc-pVTZ data
rfile.put_all('/', combined_data_ccpvtz)

# Print relative energies in kcal/mol
E_ccpvt = rfile.get('energy_ele_rimp2_ccpvtz')
E_def2tzvp = rfile.get('energy_ele_rimp2_def2tzvp')

idx_min_ccpvt = 0
yao_isomer_order = [3, 2, 4, 1]
yao_isomer_energies = [30.961317, 40.888517, 50.840818, 52.007985]

E_ccpvt_rel = E_ccpvt - E_ccpvt[idx_min_ccpvt]  # Eh
E_def2tzvp_rel = E_def2tzvp - E_def2tzvp[idx_min_ccpvt]  # Eh
E_ccpvt_rel *= hartree2kcalmol
E_def2tzvp_rel *= hartree2kcalmol


print(f'Yao et al.        {yao_isomer_energies}')
print('\nRelative energies w.r.t. yao2017many ordering')
print('---------------------------------------------')
print(f'RI-MP2/cc-pVTZ:   {E_ccpvt_rel[yao_isomer_order]}')
print(f'RI-MP2/def2-TZVP: {E_def2tzvp_rel[yao_isomer_order]}')
