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

"""Compute nearest neighbors."""

import os
import numpy as np
from scipy.integrate import trapezoid
import pandas as pd

data_dict = {
    "H2O O-O": {
        "md_path": 'analysis/md/rdf/h2o/137h2o-mbgdml-nvt_1_2-rdf-oo.npz',
        "exp_path": 'external/md/h2o-rdf/soper2013radial-oo.csv',
        "valley_search_space_md": (3.1, 3.6),  # Min then max
        "valley_search_space_exp": (3.1, 3.7),  # Min then max
    },
    "H2O O-H": {
        "md_path": 'analysis/md/rdf/h2o/137h2o-mbgdml-nvt_1_2-rdf-oh.npz',
        "exp_path": 'external/md/h2o-rdf/soper2013radial-oh.csv',
        "valley_search_space_md": (2.1, 2.7),  # Min then max
        "valley_search_space_exp": (2.1, 2.7),  # Min then max
    },
    "H2O H-H": {
        "md_path": 'analysis/md/rdf/h2o/137h2o-mbgdml-nvt_1_2-rdf-hh.npz',
        "exp_path": 'external/md/h2o-rdf/soper2013radial-hh.csv',
        "valley_search_space_md": (2.8, 3.2),  # Min then max
        "valley_search_space_exp": (2.8, 3.2),  # Min then max
    },
    "MeCN N-N": {
        "md_path": 'analysis/md/rdf/mecn/67mecn-mbgdml-nvt_1_2_3-298-rdf-nn.npz',
        "exp_path": 'external/md/mecn-rdf/humphreys2015neutron-fig6-nn.csv',
        "valley_search_space_md": (4.6, 5.1),  # Min then max
        "valley_search_space_exp": (6.0, 6.5),  # Min then max
    },
    "MeCN C-N": {
        "md_path": 'analysis/md/rdf/mecn/67mecn-mbgdml-nvt_1_2_3-298-rdf-cn.npz',
        "exp_path": 'external/md/mecn-rdf/humphreys2015neutron-fig6-cn.csv',
        "valley_search_space_md": (4.5, 5.1),  # Min then max
        "valley_search_space_exp": (4.5, 5.1),  # Min then max
    },
    "MeCN CN-CN": {
        "md_path": 'analysis/md/rdf/mecn/67mecn-mbgdml-nvt_1_2_3-298-rdf-c_n-c_n.npz',
        "exp_path": 'external/md/mecn-rdf/humphreys2015neutron-figs4b-cncn.csv',
        "valley_search_space_md": (5.9, 6.5),  # Min then max
        "valley_search_space_exp": (6.2, 6.8),  # Min then max
    },
    "MeOH O-O": {
        "md_path": 'analysis/md/rdf/meoh/61meoh-mbgdml-nvt_1_2_3-rdf-oo.npz',
        "exp_path": 'external/md/meoh-rdf/yamaguchi1999structure-erratum-oo.csv',
        "valley_search_space_md": (3.3, 3.9),  # Min then max
        "valley_search_space_exp": (3.1, 3.7),  # Min then max
    },
    "MeOH O-H": {
        "md_path": 'analysis/md/rdf/meoh/61meoh-mbgdml-nvt_1_2_3-rdf-oh.npz',
        "exp_path": 'external/md/meoh-rdf/yamaguchi1999structure-erratum-oh.csv',
        "valley_search_space_md": (2.0, 3.0),  # Min then max
        "valley_search_space_exp": (2.0, 3.0),  # Min then max
    },
    "MeOH H-H": {
        "md_path": 'analysis/md/rdf/meoh/61meoh-mbgdml-nvt_1_2_3-rdf-hh.npz',
        "exp_path": 'external/md/meoh-rdf/yamaguchi1999structure-erratum-hh.csv',
        "valley_search_space_md": (3.0, 4.0),  # Min then max
        "valley_search_space_exp": (3.0, 4.0),  # Min then max
    },
}





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../'
data_dir = os.path.join(base_dir, 'data/')

def get_n_neighbors(r_all, g_all, valley_search_bounds):
    min_idxs = np.argsort(r_all)
    r_within_valley = np.where(
        (r_all > min(valley_search_bounds)) & (r_all < max(valley_search_bounds))
    )
    min_g = np.min(g_all[r_within_valley])
    min_g_idx = np.where(g_all == min_g)[0][0]
    r_integrate = r_all[:min_g_idx+1]
    g_integrate = g_all[:min_g_idx+1]

    r_valley = r_integrate[-1]
    n_neighbors = trapezoid(g_integrate, x=r_integrate)
    return r_valley, n_neighbors

for rdf_label, rdf_info in data_dict.items():

    md_path = os.path.join(base_dir, rdf_info["md_path"])
    md_data = dict(np.load(md_path, allow_pickle=True))
    r_md = md_data['bins']
    g_md = md_data['rdf']
    r_shell_exterior_md, n_neighbors_md = get_n_neighbors(
        r_md, g_md, rdf_info["valley_search_space_md"]
    )

    exp_path = os.path.join(data_dir, rdf_info["exp_path"])
    df = pd.read_csv(exp_path)
    r_exp = df['r'].values
    g_exp = df['g'].values
    r_shell_exterior_exp, n_neighbors_exp = get_n_neighbors(
        r_exp, g_exp, rdf_info["valley_search_space_exp"]
    )

    print(rdf_label)
    print(
        f"MD      Shell limit = {r_shell_exterior_md:.2f} A"
        f"    n_neighbors = {n_neighbors_md:.2f}"
    )
    print(
        f"Exp.    Shell limit = {r_shell_exterior_exp:.2f} A"
        f"    n_neighbors = {n_neighbors_exp:.2f}"
    )
    print("\n")
