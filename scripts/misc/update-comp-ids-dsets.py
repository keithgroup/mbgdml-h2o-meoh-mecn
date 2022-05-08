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
Updates stored data set name to current file name.
"""

import os
from mbgdml.utils import get_files
from mbgdml.data import dataSet


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# file_paths = get_files('../../data/datasets', '.npz')
file_paths = get_files('/home/alex/repos/keith/mbGDML-dev/tests/data/datasets', '.npz')

for file_path in file_paths:
    file_name = os.path.basename(file_path)[:-4]  # Trims .npz
    save_dir = os.path.dirname(os.path.realpath(file_path))
    dset = dataSet(file_path)

    if dset.comp_ids.ndim == 2:
        if dset.comp_ids.shape != (1, 0):
            try:
                dset.comp_ids = dset.comp_ids[:,1]
                dset.save(file_name, dset.asdict, save_dir)
            except Exception as e:
                print(file_name)
                raise e
    
    