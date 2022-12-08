import ase
import ase.io
import csv
import numpy as np
import os

csv_name = 'model-valid-stats.csv'

os.chdir(os.path.dirname(os.path.realpath(__file__)))
valid_ref_path = 'valid.xyz'
valid_model_path = 'model-valid.xyz'
valid_ref = ase.io.read(valid_ref_path, ':')
valid_model = ase.io.read(valid_model_path, ':')

hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev

# Energy data
E_true = np.array([at.get_potential_energy() for at in valid_ref])
E_pred = np.array([at.get_potential_energy() for at in valid_model])

# Force data
F_true, F_pred = [], []
for at_ref, at_pred in zip(valid_ref, valid_model):
    F_true.append(at_ref.get_forces())
    F_pred.append(at_pred.arrays['force'])
F_true = np.array(F_true)
F_pred = np.array(F_pred)

# Convert eV to kcal/mol
E_true *= ev2kcalmol
E_pred *= ev2kcalmol
F_true *= ev2kcalmol
F_pred *= ev2kcalmol

# Get errors
E_errors = E_pred - E_true
F_errors = F_pred - F_true

# Compute error statistics
E_mae = np.nanmean(np.abs(E_errors))
E_rmse = np.sqrt(np.nanmean((E_errors)**2))
E_max_error = np.nanmax(np.abs(E_errors))
E_sse = np.dot(E_errors[~np.isnan(E_errors)], E_errors[~np.isnan(E_errors)])
F_mae = np.nanmean(np.abs(F_errors))
F_rmse = np.sqrt(np.nanmean((F_errors)**2))
F_max_error = np.nanmax(np.abs(F_errors))
F_sse = np.dot(F_errors[~np.isnan(F_errors)], F_errors[~np.isnan(F_errors)])

csv_data = [
    ['E_unit', 'R_unit', 'E_mae', 'E_rmse', 'E_sse', 'E_max_abs_err', 'F_mae', 'F_rmse', 'F_sse', 'F_max_abs_err'],
    ['kcal/mol', 'Angstrom', E_mae, E_rmse, E_sse, E_max_error, F_mae, F_rmse, F_sse, F_max_error]
]

with open(csv_name, 'w', encoding='utf-8') as f_csv:
    csv_writer = csv.writer(f_csv)
    csv_writer.writerows(csv_data)

