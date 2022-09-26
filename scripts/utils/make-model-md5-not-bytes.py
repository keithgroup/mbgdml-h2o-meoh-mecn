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
Loads and saves all datasets or structure sets to update MD5 and file name.
"""

import os
import numpy as np
from mbgdml.utils import get_files

dir_to_search = '../../data/ml-dsets'

file_paths = get_files(dir_to_search, '.npz')


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

for file_path in file_paths:
    file_name = os.path.basename(file_path)[:-4]  # Trims .npz
    save_dir = os.path.dirname(os.path.realpath(file_path))
    npz_dict = dict(np.load(file_path, allow_pickle=True))
    try:
        npz_type = npz_dict['type'].item()
    except KeyError:
        continue
    if npz_type != 'm':
        continue

    md5 = npz_dict['md5'].item().decode("utf-8")
    npz_dict['md5'] = np.array(md5)

    md5_train = npz_dict['md5_train'].item().decode("utf-8")
    npz_dict['md5_train'] = np.array(md5_train)

    md5_valid = npz_dict['md5_valid'].item().decode("utf-8")
    npz_dict['md5_valid'] = np.array(md5_valid)

    try:
        md5_test = npz_dict['md5_test'].item().decode("utf-8")
        npz_dict['md5_test'] = np.array(md5_test)
    except AttributeError:
        pass
    
    print(npz_dict['md5'])

    np.savez_compressed(file_path, **npz_dict)
    
    