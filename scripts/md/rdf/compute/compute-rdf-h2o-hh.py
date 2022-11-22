#!/usr/bin/env python3

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
Computes the H-H rdf curve in pure water.
"""

import os
import numpy as np
from mbgdml.analysis.rdf import RDF
from reptar import File
import matplotlib.pyplot as plt

exdir_path = 'data/md/h2o/137h2o-mbgdml-md.exdir'
example_key = '1-nvt'
R_path = 'data/md/h2o/137h2o-mbgdml-nvt_1_2.npy'

save_path = 'analysis/md/rdf/h2o/137h2o-mbgdml-nvt_1_2-rdf-hh.npz'

comp_id_pair = ('h2o', 'h2o')
entity_idxs = ((1, 2), (1, 2))  # 8 1 1
bin_width = 0.1  # Ang
rdf_range = (0.0, 8.0)  # lower and upper bound in Ang

start_frame = 4590
step = 1
n_workers = 8  # None or 8


###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../../'
exdir_path = os.path.join(base_dir, exdir_path)
R_path = os.path.join(base_dir, R_path)
save_path = os.path.join(base_dir, save_path)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

print('Getting data')
rfile = File(exdir_path)
Z = rfile.get(f'{example_key}/atomic_numbers')
R = np.load(R_path, mmap_mode='r')
entity_ids = rfile.get(f'{example_key}/entity_ids')
comp_ids = rfile.get(f'{example_key}/comp_ids')
cell_vectors = np.array(rfile.get(f'{example_key}/periodic_cell'))

print('Computing rdf')
rdf = RDF(
    Z, entity_ids, comp_ids, cell_vectors, bin_width=bin_width,
    rdf_range=rdf_range, inter_only=True, n_workers=n_workers
)
rdf._max_chunk_size = 10
bins, gr = rdf.run(R[start_frame:], comp_id_pair, entity_idxs, step=step)

print('Saving results')
npz_dict = {
    'bins': bins,
    'rdf': gr,
    'start_frame': start_frame
}
np.savez(save_path, **npz_dict)

