#!/usr/bin/env python3

# MIT License
# 
# Copyright (c) 2021, Alex M. Maldonado
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
Performs a path similarity analysis with MDAnalysis.

API information: https://userguide.mdanalysis.org/stable/examples/analysis/trajectory_similarity/psa.html
Literature reference: https://doi.org/10.1371/journal.pcbi.1004568
"""

import os
import MDAnalysis as mda
from MDAnalysis.analysis import psa
import sys
sys.setrecursionlimit(5000)  # Needed for discrete Fréchet distances

metric = 'discrete_frechet'  # hausdorff or discrete_frechet

ref_pdb_path = 'h2o/6h2o/orca/6h2o-ase.md-orca.mp2.def2tzvp.300k-0.pdb'
pdb_paths = [
    'h2o/6h2o/gdml/6h2o-ase.md-mbgdml.train1000.300k-0/6h2o-ase.md-mbgdml.train1000.300k-0.pdb',
    'h2o/6h2o/gap/6h2o-ase.md-gap.train1000.300k-0/6h2o-ase.md-gap.train1000.300k-0.pdb',
    'h2o/6h2o/schnet/6h2o-ase.md-schnet.train1000.300k-0/6h2o-ase.md-schnet.train1000.300k-0.pdb',
    'h2o/6h2o/gfn2/6h2o-ase.md-gfn2.300k-0/6h2o-ase.md-gfn2.300k-0.pdb',
    'h2o/6h2o/orca/6h2o-ase.md-orca.mp2.def2svp.300k-0/6h2o-ase.md-orca.mp2.def2svp.300k-0.pdb',
    'h2o/6h2o/orca/6h2o-ase.md-orca.rimp2.def2tzvp.300k-0/6h2o-ase.md-orca.rimp2.def2tzvp.300k-0.pdb',
]
labels = ['mbGDML', 'mbGAP', 'mbSchNet', 'gfn2', 'MP2/def2-SVP', 'RI-MP2/def2-TZVP']

###   SCRIPT   ###
base_dir = '/home/alex/Dropbox/keith/projects/mbgdml-h2o-meoh-mecn/data/md'
ref_pdb_path = os.path.join(base_dir, ref_pdb_path)
pdb_paths = [os.path.join(base_dir, pdb_path) for pdb_path in pdb_paths]

ref = mda.Universe(ref_pdb_path, dt=0.001)
us = [mda.Universe(pdb_path, dt=0.001) for pdb_path in pdb_paths]
us.insert(0, ref)

ps = psa.PSAnalysis(
    us,
    labels=labels,
    select='chainID A'
)

ps.generate_paths(align=True, save=False, weights=None)

ps.run(metric=metric)

dists = ps.D[:,0][1:]
for i in range(len(dists)):
    print(f'{metric} with {labels[i]}: {dists[i]:.3f}')
