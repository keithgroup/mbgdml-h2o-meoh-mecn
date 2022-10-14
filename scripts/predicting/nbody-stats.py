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
Compute error statistics for n-body predictions.
"""

import os
import numpy as np
from mbgdml.mbe import mbe_contrib
from reptar import File


systems_dict = {
    'H2O 4-6mers': {
        'rfile_path': 'h2o-temelso.etal.exdir',
        'e_ref_key': 'energy_ele_mp2.def2tzvp_orca',
        'g_ref_key': 'grads_mp2.def2tzvp_orca',
        'group_keys': {
            '4h2o': {
                'nbody_keys': ['samples_1h2o', 'samples_2h2o', 'samples_3h2o'],
            },
            '5h2o': {
                'nbody_keys': ['samples_1h2o', 'samples_2h2o', 'samples_3h2o'],
            },
            '6h2o': {
                'nbody_keys': ['samples_1h2o', 'samples_2h2o', 'samples_3h2o'],
            },
        }
    },
    'H2O 16mer': {
        'rfile_path': '16h2o-yoo.etal.exdir',
        'e_ref_key': 'energy_ele_rimp2.def2tzvp_orca',
        'g_ref_key': 'grads_rimp2.def2tzvp_orca',
        'group_keys': {
            '': {
                'nbody_keys': ['samples_1h2o', 'samples_2h2o', 'samples_3h2o'],
            },
        }
    },
    'MeCN 4-6mers': {
        'rfile_path': 'mecn-malloum.etal.exdir',
        'e_ref_key': 'energy_ele_mp2.def2tzvp_orca',
        'g_ref_key': 'grads_mp2.def2tzvp_orca',
        'group_keys': {
            '4mecn': {
                'nbody_keys': ['samples_1mecn', 'samples_2mecn', 'samples_3mecn'],
            },
            '5mecn': {
                'nbody_keys': ['samples_1mecn', 'samples_2mecn', 'samples_3mecn'],
            },
            '6mecn': {
                'nbody_keys': ['samples_1mecn', 'samples_2mecn', 'samples_3mecn'],
            },
        }
    },
    'MeCN 16mer': {
        'rfile_path': '16mecn-remya.etal.exdir',
        'e_ref_key': 'energy_ele_rimp2.def2tzvp_orca',
        'g_ref_key': 'grads_rimp2.def2tzvp_orca',
        'group_keys': {
            '': {
                'nbody_keys': ['samples_1mecn', 'samples_2mecn', 'samples_3mecn'],
            },
        }
    },
    'MeOH 4-6mers': {
        'rfile_path': 'meoh-boyd.etal.exdir',
        'e_ref_key': 'energy_ele_mp2.def2tzvp_orca',
        'g_ref_key': 'grads_mp2.def2tzvp_orca',
        'group_keys': {
            '4meoh': {
                'nbody_keys': ['samples_1meoh', 'samples_2meoh', 'samples_3meoh'],
            },
            '5meoh': {
                'nbody_keys': ['samples_1meoh', 'samples_2meoh', 'samples_3meoh'],
            },
            '6meoh': {
                'nbody_keys': ['samples_1meoh', 'samples_2meoh', 'samples_3meoh'],
            },
        }
    },
    'MeOH 16mer': {
        'rfile_path': '16meoh-pires.deturi.exdir',
        'e_ref_key': 'energy_ele_rimp2.def2tzvp_orca',
        'g_ref_key': 'grads_rimp2.def2tzvp_orca',
        'group_keys': {
            '/': {
                'nbody_keys': ['samples_1meoh', 'samples_2meoh', 'samples_3meoh'],
            },
        }
    },
}

e_key = 'energy_ele_mp2.def2tzvp_orca'
e_mb_key = 'energy_ele_nbody_mp2.def2tzvp_orca'
g_key = 'grads_mp2.def2tzvp_orca'
g_mb_key = 'grads_nbody_mp2.def2tzvp_orca'







###   SCRIPT   ###
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
data_dir = os.path.join(base_dir, 'data/isomers/')

def get_mb_ef(group_key):
    global rfile
    try:
        E = rfile.get(f'{group_key}/{e_mb_key}')
    except RuntimeError:
        E = rfile.get(f'{group_key}/{e_key}')
    try:
        G = rfile.get(f'{group_key}/{g_mb_key}')
    except RuntimeError:
        G = rfile.get(f'{group_key}/{g_key}')
    return E, -G

print('All energies in kcal/mol\n')
for label, data_dict in systems_dict.items():
    print(label)

    rfile_path = os.path.join(data_dir, data_dict['rfile_path'])
    rfile = File(rfile_path, mode='r')

    e_ref_key = data_dict['e_ref_key']
    g_ref_key = data_dict['g_ref_key']

    E_errors = []
    E_errors_per_monomer = []
    F_errors = []
    F_errors_per_atom = []
    group_dict = data_dict['group_keys']
    for group_key,group_data in group_dict.items():
        Z = rfile.get(f'{group_key}/atomic_numbers')
        entity_ids = rfile.get(f'{group_key}/entity_ids')
        


        n_atoms = int(len(Z))
        n_monomers = int(len(set(entity_ids)))
        
        E_true = rfile.get(f'{group_key}/{e_ref_key}')
        if E_true.shape == ():
            E_true = np.array([E_true])
        F_true = -rfile.get(f'{group_key}/{g_ref_key}')

        # Make many-body predictions
        E_pred = np.zeros(E_true.shape)
        F_pred = np.zeros(F_true.shape)
        nbody_keys = group_data['nbody_keys']
        for nbody_key in nbody_keys:
            nbody_key = f'{group_key}/{nbody_key}'
            E_lower, F_lower = get_mb_ef(nbody_key)
            entity_ids_lower = rfile.get(f'{nbody_key}/entity_ids')
            r_prov_ids_lower = rfile.get(f'{nbody_key}/r_prov_ids')
            r_prov_specs_lower = rfile.get(f'{nbody_key}/r_prov_specs')
            
            E_pred, F_pred = mbe_contrib(
                E_pred, F_pred, entity_ids, None, None, E_lower, F_lower,
                entity_ids_lower, r_prov_ids_lower, r_prov_specs_lower,
                operation='add'
            )

        E_error = E_pred - E_true  # Eh
        E_error *= hartree2kcalmol
        F_error = np.ravel(F_pred - F_true)
        F_error *= hartree2kcalmol

        E_error_per_monomer = E_error/n_monomers
        F_error_per_atom = F_error/n_atoms

        E_errors.extend(E_error.tolist())
        E_errors_per_monomer.extend(E_error_per_monomer.tolist())
        F_errors.extend(F_error.tolist())
        F_errors_per_atom.extend(F_error_per_atom.tolist())

    E_errors = np.array(E_errors)
    E_errors_per_monomer = np.array(E_errors_per_monomer)
    F_errors = np.array(F_errors)
    F_errors_per_atom = np.array(F_errors_per_atom)

    E_mae = np.mean(np.abs(E_errors))
    E_rmse = np.sqrt(np.mean((E_errors)**2))
    E_mae_per_monomer = np.mean(np.abs(E_errors_per_monomer))
    E_rmse_per_monomer = np.sqrt(np.mean((E_errors_per_monomer)**2))

    F_mae = np.mean(np.abs(F_errors))
    F_rmse = np.sqrt(np.mean((F_errors)**2))
    F_mae_per_atom = np.mean(np.abs(F_errors_per_atom))
    F_rmse_per_atom = np.sqrt(np.mean((F_errors_per_atom)**2))
    
    print('-----------------')
    print(f'Energy MAE/RMSE                 {E_mae:.3f} / {E_rmse:.3f}')
    print(f'Energy per monomer MAE/RMSE     {E_mae_per_monomer:.3f} / {E_rmse_per_monomer:.3f}')

    print(f'Force MAE/RMSE                  {F_mae:.3f} / {F_rmse:.3f}')
    print(f'Force per atom MAE/RMSE         {F_mae_per_atom:.3f} / {F_rmse_per_atom:.3f}')
    print()