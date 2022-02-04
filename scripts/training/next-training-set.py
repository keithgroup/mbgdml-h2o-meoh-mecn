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
Determines the next training set.
"""

import os
from shutil import copyfile
import subprocess
from mbgdml.data import mbModel

run_mlff_path = '/home/alex/repos/MLFF/run.py'
para_file_path = '/home/alex/repos/MLFF/paras/default.py'

# Data set directories.
dset_dir = '../../data/datasets'
dset_dir_h2o = f'{dset_dir}/h2o'
dset_dir_mecn = f'{dset_dir}/mecn'
dset_dir_meoh = f'{dset_dir}/meoh'

# Model dirs.
model_dir = '../../data/models'
model_dir_h2o = f'{model_dir}/h2o'
model_dir_mecn = f'{model_dir}/mecn'
model_dir_meoh = f'{model_dir}/meoh'

# Cluster_error directories.
cluster_error_dir = '../../analysis/iterative-training'
cluster_error_h2o = f'{cluster_error_dir}/h2o'
cluster_error_mecn = f'{cluster_error_dir}/mecn'
cluster_error_meoh = f'{cluster_error_dir}/meoh'

dset_dir_solvent = dset_dir_meoh
model_dir_solvent = model_dir_meoh
cluster_error_solvent = cluster_error_meoh

dset_path = f'{dset_dir_solvent}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh-dset.mb-cm14.npz'
model_path = f'{model_dir_solvent}/3meoh/nosym/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain900.nosym.npz'
random200_cluster_path = f'{cluster_error_solvent}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-randomtrain200/saves/cluster_error_62meoh/cluster_indices.npy'

model_file_name = os.path.splitext(os.path.basename(model_path))[0]
save_dir = f'{cluster_error_solvent}/3meoh/{model_file_name}'

# If this is the first iterative training clustering (won't load cluster file)
first_run = False
cluster_error = True
get_new_train = True

# Number of new training set structures.
# Sometimes this needs to be manually changed to get the right amount (printed at the end).
add_n_train = 100




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Runs cluster_error script.
para_dir = save_dir + '/paras'
os.makedirs(save_dir, exist_ok=True)
os.chdir(save_dir)

# Need to copy over parameter file.
os.makedirs(para_dir, exist_ok=True)
copyfile(para_file_path, para_dir + '/default.py')

if cluster_error:
    cluster_error_command = f"python {run_mlff_path} cluster_error -d {dset_path} -i {model_path}"
    if not first_run:
        cluster_error_command += f' -c {random200_cluster_path}'
    process = subprocess.run(cluster_error_command.split())

if get_new_train:
    print('\n\n\n\n')
    new_train_command = f"python {run_mlff_path} train -d {dset_path} -i {model_path} -c {random200_cluster_path} -s {add_n_train} -n 1"
    process = subprocess.run(new_train_command.split())