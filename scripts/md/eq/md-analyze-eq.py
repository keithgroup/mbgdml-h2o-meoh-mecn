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

"""Analyze MD trajectory with pymbar."""

import os
from reptar import File
from pymbar import timeseries

exdir_path = 'h2o/137h2o-mbgdml-md.exdir'
group_key = '1-nvt'



###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

base_dir = '../../../data/md'
exdir_path = os.path.join(base_dir, exdir_path)

rfile = File(exdir_path)

E = rfile.get(f'{group_key}/energy_pot')
temp = rfile.get(f'{group_key}/temp')

t0_E, g_E, Neff_max_E = timeseries.detect_equilibration(E, nskip=10)
t0_temp, g_temp, Neff_max_temp = timeseries.detect_equilibration(temp, nskip=10)

print(f'Start of prod: {max(t0_E, t0_E)}')

