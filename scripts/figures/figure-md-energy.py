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
Analyzes difference in MP2/def2-TZVP MD electronic energies to mbGDML
predictions of same structures."""

import numpy as np
import matplotlib.pyplot as plt
from mbgdml.data import predictSet


# Predict set dirs.
predictset_dir = '../../data/predictsets'
predictset_dir_h2o = f'{predictset_dir}/h2o'
predictset_dir_mecn = f'{predictset_dir}/mecn'
predictset_dir_meoh = f'{predictset_dir}/meoh'

predictset_dir_solvent = predictset_dir_meoh
pset_path = f'{predictset_dir_solvent}/6meoh/6meoh.boyd.etal.1.md.orca.mp2.def2tzvp.300k-pset-62meoh.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'

save_plot = False
num_monomers = 6



save = True
save_dir = '../../analysis/md/md-energies'

# Plot

solvent_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}
cluster_num_color = 'silver'



###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

if save_dir[-1] != '/':
    save_dir += '/'

pset = predictSet(pset_path)
E_true = pset.E_true
E, F = pset.nbody_predictions([1, 2, 3])

E_error = E - E_true
E_error_per_monomer = E_error/num_monomers

mae_per_monomer = np.mean(np.abs(E_error_per_monomer))
print(f'Energy MAE {mae_per_monomer:.3f} kcal/mol')

