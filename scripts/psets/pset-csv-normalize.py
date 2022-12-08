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
Prints normalized statistics from predictset csv files.

We normalize energies by the number of entities (i.e., molecules) and forces by atom.
"""

import os
import numpy as np
from mbgdml.data import DataSet
import pandas as pd


csv_path = 'meoh/meoh-psets-schnet.niter5.nfeat128.best.train1000.csv'
atoms_per_entity = 6
entities_per_row = [  # Same for each CSV file
    1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 6, 1, 2, 3, 16, 1, 2, 3
]

drop_columns = [
    '1-body', '2-body', '3-body', 'E_unit', 'R_unit', 'E_sse', 'F_sse', 'time',
    'F_max_abs_err', 'E_max_abs_err'
]


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../'
data_dir = 'data/psets'
csv_path = os.path.join(base_dir, data_dir, csv_path)

df = pd.read_csv(csv_path, header=0)
df.columns = df.columns.str.replace(' ', '')

df = df.drop(drop_columns, axis=1)
print('Total statistics')
print(df)
atoms_per_row = [i*atoms_per_entity for i in entities_per_row]
entities_per_row = np.array(entities_per_row)
atoms_per_row = np.array(atoms_per_row)

df['E_mae'] = np.divide(df['E_mae'], entities_per_row)
df['E_rmse'] = np.divide(df['E_rmse'], entities_per_row)
df['F_mae'] = np.divide(df['F_mae'], atoms_per_row)
df['F_rmse'] = np.divide(df['F_rmse'], atoms_per_row)

print('\n\nNormalized statistics')
print(df)
