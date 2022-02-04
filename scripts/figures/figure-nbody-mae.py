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
Analyzes predictSets and prints energy and force MAEs. Lolliplots can be generated
for energy MAEs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mbgdml.data import predictSet

###   SCRIPT   ###

# Predict set paths.
pset_dir = '../../data/predictsets'
pset_dir_h2o = f'{pset_dir}/h2o'
pset_dir_mecn = f'{pset_dir}/mecn'
pset_dir_meoh = f'{pset_dir}/meoh'

solvents = ['h2o', 'mecn', 'meoh']

train_type = 'iterative'

include_forces = True  # True if you want force RMSE 

save_figure = False

if train_type == 'random':
    isomer_psets = {
        'h2o': {
            1: [
                #predictSet(f'{pset_dir_h2o}/1h2o/4h2o.temelso.etal.dset.1h2o-pset-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.1h2o.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/1h2o/5h2o.temelso.etal.dset.1h2o-pset-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.1h2o.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/1h2o/6h2o.temelso.etal.dset.1h2o-pset-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.1h2o.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/1h2o/16h2o.yoo.etal.boat.b.1h2o-pset-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.1h2o.randomtrain1000.npz'),
                predictSet(f'{pset_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.1h2o.randomtrain1000.npz'),
            ],
            2: [
                #predictSet(f'{pset_dir_h2o}/2h2o/4h2o.temelso.etal.dset.2h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/2h2o/5h2o.temelso.etal.dset.2h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/2h2o/6h2o.temelso.etal.dset.2h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/2h2o/16h2o.yoo.etal.boat.b.2h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.randomtrain1000.npz'),
                predictSet(f'{pset_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.randomtrain1000.npz'),
            ],
            3: [
                #predictSet(f'{pset_dir_h2o}/3h2o/4h2o.temelso.etal.dset.3h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/3h2o/5h2o.temelso.etal.dset.3h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/3h2o/6h2o.temelso.etal.dset.3h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/3h2o/16h2o.yoo.etal.boat.b.3h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.randomtrain1000.npz'),
                predictSet(f'{pset_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.randomtrain1000.npz'),
            ],
        },
        'mecn': {
            1: [
                #predictSet(f'{pset_dir_mecn}/1mecn/4mecn.malloum.etal.dset.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/1mecn/5mecn.malloum.etal.dset.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/1mecn/6mecn.malloum.etal.dset.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/1mecn/16mecn.remya.etal.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.randomtrain1000.npz'),
                predictSet(f'{pset_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.randomtrain1000.npz'),
            ],
            2: [
                #predictSet(f'{pset_dir_mecn}/2mecn/4mecn.malloum.etal.dset.2mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/2mecn/5mecn.malloum.etal.dset.2mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/2mecn/6mecn.malloum.etal.dset.2mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/2mecn/16mecn.remya.etal.2mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.randomtrain1000.npz'),
                predictSet(f'{pset_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm.17.dset.2mecn.cm9-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.randomtrain1000.npz'),
            ],
            3: [
                #predictSet(f'{pset_dir_mecn}/3mecn/4mecn.malloum.etal.dset.3mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/3mecn/5mecn.malloum.etal.dset.3mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/3mecn/6mecn.malloum.etal.dset.3mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/3mecn/16mecn.remya.etal.3mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.randomtrain1000.npz'),
                predictSet(f'{pset_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.randomtrain1000.npz'),
            ]
        },
        'meoh': {
            1: [
                predictSet(f'{pset_dir_meoh}/1meoh/4meoh.boyd.etal.dset.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/1meoh/5meoh.boyd.etal.dset.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/1meoh/6meoh.boyd.etal.dset.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/1meoh/16meoh.pires.deturi.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.randomtrain1000.npz'),
            ],
            2: [
                predictSet(f'{pset_dir_meoh}/2meoh/4meoh.boyd.etal.dset.2meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/2meoh/5meoh.boyd.etal.dset.2meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/2meoh/6meoh.boyd.etal.dset.2meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/2meoh/16meoh.pires.deturi.2meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.randomtrain1000.npz'),
            ],
            3: [
                predictSet(f'{pset_dir_meoh}/3meoh/4meoh.boyd.etal.dset.3meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/3meoh/5meoh.boyd.etal.dset.3meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/3meoh/6meoh.boyd.etal.dset.3meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/3meoh/16meoh.pires.deturi.3meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.randomtrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.randomtrain1000.npz'),
            ]
        }
    }
    plot_name = 'nbody-errors-randomtrain1000-energies'
    if include_forces:
        plot_name +='-and-forces'
elif train_type == 'iterative':
    isomer_psets = {
        'h2o': {
            1: [
                #predictSet(f'{pset_dir_h2o}/1h2o/4h2o.temelso.etal.dset.1h2o-pset-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/1h2o/5h2o.temelso.etal.dset.1h2o-pset-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/1h2o/6h2o.temelso.etal.dset.1h2o-pset-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/1h2o/16h2o.yoo.etal.boat.b.1h2o-pset-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o.iterativetrain1000.npz'),
                predictSet(f'{pset_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-pset-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o.iterativetrain1000.npz'),
            ],
            2: [
                #predictSet(f'{pset_dir_h2o}/2h2o/4h2o.temelso.etal.dset.2h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/2h2o/5h2o.temelso.etal.dset.2h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/2h2o/6h2o.temelso.etal.dset.2h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/2h2o/16h2o.yoo.etal.boat.b.2h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.iterativetrain1000.npz'),
                predictSet(f'{pset_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6.iterativetrain1000.npz'),
            ],
            3: [
                #predictSet(f'{pset_dir_h2o}/3h2o/4h2o.temelso.etal.dset.3h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/3h2o/5h2o.temelso.etal.dset.3h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/3h2o/6h2o.temelso.etal.dset.3h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_h2o}/3h2o/16h2o.yoo.etal.boat.b.3h2o-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.iterativetrain1000.npz'),
                predictSet(f'{pset_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-pset.mb-140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.iterativetrain1000.npz'),
            ],
        },
        'mecn': {
            1: [
                #predictSet(f'{pset_dir_mecn}/1mecn/4mecn.malloum.etal.dset.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/1mecn/5mecn.malloum.etal.dset.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/1mecn/6mecn.malloum.etal.dset.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/1mecn/4mecn.malloum.etal.dset.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.iterativetrain1000.npz'),
                predictSet(f'{pset_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-pset-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn.iterativetrain1000.npz'),
            ],
            2: [
                #predictSet(f'{pset_dir_mecn}/2mecn/4mecn.malloum.etal.dset.2mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/2mecn/5mecn.malloum.etal.dset.2mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/2mecn/6mecn.malloum.etal.dset.2mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/2mecn/16mecn.remya.etal.2mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.iterativetrain1000.npz'),
                predictSet(f'{pset_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9.iterativetrain1000.npz'),
            ],
            3: [
                #predictSet(f'{pset_dir_mecn}/3mecn/4mecn.malloum.etal.dset.3mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/3mecn/5mecn.malloum.etal.dset.3mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/3mecn/6mecn.malloum.etal.dset.3mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_mecn}/3mecn/16mecn.remya.etal.3mecn-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.iterativetrain1000.npz'),
                predictSet(f'{pset_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-pset.mb-48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.iterativetrain1000.npz'),
            ]
        },
        'meoh': {
            1: [
                #predictSet(f'{pset_dir_meoh}/1meoh/4meoh.boyd.etal.dset.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/1meoh/5meoh.boyd.etal.dset.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/1meoh/6meoh.boyd.etal.dset.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/1meoh/16meoh.pires.deturi.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.iterativetrain1000.npz'),
                predictSet(f'{pset_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-pset-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh.iterativetrain1000.npz'),
            ],
            2: [
                #predictSet(f'{pset_dir_meoh}/2meoh/4meoh.boyd.etal.dset.2meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/2meoh/5meoh.boyd.etal.dset.2meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/2meoh/6meoh.boyd.etal.dset.2meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/2meoh/16meoh.pires.deturi.2meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.iterativetrain1000.npz'),
                predictSet(f'{pset_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8.iterativetrain1000.npz'),
            ],
            3: [
                #predictSet(f'{pset_dir_meoh}/3meoh/4meoh.boyd.etal.dset.3meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/3meoh/5meoh.boyd.etal.dset.3meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/3meoh/6meoh.boyd.etal.dset.3meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.iterativetrain1000.npz'),
                #predictSet(f'{pset_dir_meoh}/3meoh/16meoh.pires.deturi.3meoh-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.iterativetrain1000.npz'),
                predictSet(f'{pset_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-pset.mb-62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.iterativetrain1000.npz'),
            ]
        }
    }
    plot_name = 'nbody-errors-iterativetrain1000-energies'
    if include_forces:
        plot_name +='-and-forces'

save_dir = '../../analysis/isomer-predictions'

if save_dir[-1] != '/':
    save_dir += '/'

solvent_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}
ref_color = 'silver'





###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# FIGURE #

E_mae = []
#F_rmse = []
F_mae = []
E_num_r = []
E_solvents = []
E_colors = []
E_orders = []
E_xlabels = []

for solv in solvents:
    solv_psets = isomer_psets[solv]

    if solv == 'h2o':
        solv_label = 'H2O'
    elif solv == 'mecn':
        solv_label = 'MeCN'
    elif solv == 'meoh':
        solv_label = 'MeOH'

    for nbody_order in solv_psets:
        # Calculates mean absolute error for this n-body order.
        E_errors = np.array([])  # Absolute error.
        F_errors = np.array([[[]]])  # Error.
        for pset in solv_psets[nbody_order]:
            E_true = pset.E_true
            F_true = pset.F_true
            E_pred, F_pred = pset.nbody_predictions([nbody_order])
            E_errors = np.concatenate((E_errors, np.abs(E_pred - E_true)))
            F_error = F_pred - F_true
            if F_errors.shape == (1, 1, 0):
                F_errors = F_error
            else:
                F_errors = np.concatenate((F_errors, F_error), axis=0)

        # Adding information
        mae = np.nanmean(E_errors)
        E_mae.append(mae)  # Ignore NaN
        E_solvents.append(solv)
        E_orders.append(nbody_order)

        # Flattens and removes NaN values in forces.
        F_errors = F_errors[np.logical_not(np.isnan(F_errors))].flatten()
        #f_rmse = np.sqrt(np.mean((F_errors)**2))
        #F_rmse.append(f_rmse)
        f_mae = np.mean(np.abs(F_errors))
        F_mae.append(f_mae)

        num_r = 0
        for i in range(len(E_errors)):
            if not np.isnan(E_errors[i]):
                num_r += 1
        E_num_r.append(num_r)

        E_colors.append(solvent_colors[solv])
        E_xlabels.append(f'{nbody_order}-body\n{solv_label}')
        print(f'Energy MAE for {E_errors.shape[0]} {solv_label} {nbody_order}-body structures: {mae:.4f} kcal/mol')
        #print(f'Force RMSE: {f_rmse:.3f} kcal/mol/A\n')
        print(f'Force MAE: {f_mae:.4f} kcal/mol/A\n')


# Add data to plot.

if save_figure:
    # Setting up general figure properties
    font = {'family' : 'sans-serif',
            'size'   : 8}
    plt.rc('font', **font)

    marker_size = 5
    line_width = 2.5

    if not include_forces:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)

        # ONLY ENERGIES
        for i in range(len(E_mae)):
            line_x = [i, i]
            line_y = [0, E_mae[i]]
            marker_x = i
            marker_y = E_mae[i]
            
            ax.plot(
                line_x, line_y,
                color=E_colors[i],
                marker='', markersize=0, 
                linestyle='-', linewidth=line_width,
            )
            ax.plot(
                marker_x, marker_y,
                color=E_colors[i],
                marker='o', markersize=marker_size, 
                linestyle='', linewidth=0,
            )

        # x-axis
        ax.set_xticks([i for i in range(len(E_mae))])
        ax.set_xticklabels(E_xlabels)

        # y-axis
        ax.set_ylabel('Energy MAE (kcal/mol)')
        #y_tick_start, y_tick_end = ax.get_ylim()
        #ax.set_yticks(np.arange(0, y_tick_end, 1))
        #ax.set_yticks(np.arange(0, y_tick_end, 0.5), minor=True)
        ax.set_ylim(ymin=0)
    else:
        # FORCES AND ENERGIES
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)
        ax2 = ax1.twinx()

        i_offset = 0.1
        
        for i in range(len(E_mae)):
            # Energy
            e_x_value = i - i_offset
            e_y_value = E_mae[i]

            e_line_x = [e_x_value, e_x_value]
            e_line_y = [0, e_y_value]
            e_marker_x = e_x_value
            e_marker_y = e_y_value
            
            ax1.plot(
                e_line_x, e_line_y,
                color=E_colors[i],
                marker='', markersize=0, 
                linestyle='-', linewidth=line_width,
            )
            ax1.plot(
                e_marker_x, e_marker_y,
                color=E_colors[i],
                marker='o', markersize=marker_size, 
                linestyle='', linewidth=0,
            )

            # Forces
            f_x_value = i + i_offset
            #f_y_value = F_rmse[i]
            f_y_value = F_mae[i]

            f_line_x = [f_x_value, f_x_value]
            f_line_y = [0, f_y_value]
            f_marker_x = f_x_value
            f_marker_y = f_y_value
            
            ax2.plot(  # Outer color
                f_line_x, f_line_y,
                color=E_colors[i],
                marker='', markersize=0, 
                linestyle='-', linewidth=line_width,
            )
            ax2.plot(
                f_marker_x, f_marker_y,
                color='white',
                marker='s', markersize=marker_size - 0.5, 
                linestyle='', linewidth=0,
                markeredgewidth=1.5, markeredgecolor=E_colors[i]
            )

        # x-axis
        ax1.set_xticks([i for i in range(len(E_mae))])
        ax1.set_xticklabels(E_xlabels)

        # y-axis
        ax1.set_ylabel('Energy MAE (kcal/mol)')
        y_tick_start, y_tick_end = ax1.get_ylim()
        ax1.set_yticks(np.arange(0, y_tick_end, 0.01))
        ax1.set_yticks(np.arange(0, y_tick_end, 0.005), minor=True)
        ax1.set_ylim(ymin=0)

        #ax2.set_ylabel('Force RMSE (kcal/(mol $\AA$))')
        ax2.set_ylabel('Force MAE (kcal/(mol $\AA$))')
        y_tick_start, y_tick_end = ax2.get_ylim()
        ax2.set_yticks(np.arange(0, y_tick_end, 0.1))
        ax2.set_yticks(np.arange(0, y_tick_end, 0.05), minor=True)
        ax2.set_ylim(ymin=0)

        # Manual legend
        legend_elements = [
            Line2D(
                [0], [0], marker='o', color='darkgrey', label='Energy MAE',
                markersize=marker_size, linestyle='', linewidth=0,
            ),
            Line2D(
                #[0], [0], marker='s', color='white', label='Force RMSE',
                [0], [0], marker='s', color='white', label='Force MAE',
                markersize=marker_size - 0.5,
                markeredgewidth=1.5, markeredgecolor='darkgrey',
                linestyle='', linewidth=0,
            )
        ]
        ax1.legend(handles=legend_elements, loc='best', frameon=False)

        

    plt_path = f'{save_dir}{plot_name}.png'

    print(f'Saving {plt_path}')
    plt.savefig(plt_path, dpi=1000)

