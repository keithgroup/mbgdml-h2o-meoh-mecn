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
Ranks isomer energies containing four to six monomers and plots their
relative energies
"""

from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from mbgdml.data import dataSet, predictSet
from mbgdml.utils import e_f_contribution
from cclib.parser.utils import convertor

solvents = ['h2o']  # 'h2o', 'mecn', 'meoh'




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Plot names
plot_names = {
    'h2o': '4-6h2o.temelso.etal.isomers-iterativetrain1000-basis',
    'mecn': '4-6mecn.malloum.etal.isomers-iterativetrain1000-basis',
    'meoh': '4-6meoh.boyd.etal.isomers-iterativetrain1000-basis',
}

# Data set paths.
dset_dir = '../../data/datasets'
dset_dir_h2o = f'{dset_dir}/h2o'
dset_dir_mecn = f'{dset_dir}/mecn'
dset_dir_meoh = f'{dset_dir}/meoh'

isomer_dsets = {
    'def2tzvp': {
        'h2o': [
            dataSet(f'{dset_dir_h2o}/4h2o/4h2o.temelso.etal-dset.npz'),
            dataSet(f'{dset_dir_h2o}/5h2o/5h2o.temelso.etal-dset.npz'),
            dataSet(f'{dset_dir_h2o}/6h2o/6h2o.temelso.etal-dset.npz')
        ],
        'mecn': [
            dataSet(f'{dset_dir_mecn}/4mecn/4mecn.malloum.etal-dset.npz'),
            dataSet(f'{dset_dir_mecn}/5mecn/5mecn.malloum.etal-dset.npz'),
            dataSet(f'{dset_dir_mecn}/6mecn/6mecn.malloum.etal-dset.npz')
        ],
        'meoh': [
            dataSet(f'{dset_dir_meoh}/4meoh/4meoh.boyd.etal-dset.npz'),
            dataSet(f'{dset_dir_meoh}/5meoh/5meoh.boyd.etal-dset.npz'),
            dataSet(f'{dset_dir_meoh}/6meoh/6meoh.boyd.etal-dset.npz')
        ]
    },
    'augccpvtz': {
        'h2o': [
            dataSet(f'{dset_dir_h2o}/4h2o/4h2o.temelso.etal-dset-mp2.augccpvtz.npz'),
            dataSet(f'{dset_dir_h2o}/5h2o/5h2o.temelso.etal-dset-mp2.augccpvtz.npz'),
            dataSet(f'{dset_dir_h2o}/6h2o/6h2o.temelso.etal-dset-mp2.augccpvtz.npz')
        ],
        'mecn': [
            dataSet(f'{dset_dir_mecn}/4mecn/4mecn.malloum.etal-dset.npz'),
            dataSet(f'{dset_dir_mecn}/5mecn/5mecn.malloum.etal-dset.npz'),
            dataSet(f'{dset_dir_mecn}/6mecn/6mecn.malloum.etal-dset.npz')
        ],
        'meoh': [
            dataSet(f'{dset_dir_meoh}/4meoh/4meoh.boyd.etal-dset.npz'),
            dataSet(f'{dset_dir_meoh}/5meoh/5meoh.boyd.etal-dset.npz'),
            dataSet(f'{dset_dir_meoh}/6meoh/6meoh.boyd.etal-dset.npz')
        ]
    }
}

isomer_mb_dsets = {
    'def2tzvp': {
        'h2o': [
            [
                dataSet(f'{dset_dir_h2o}/1h2o/temelso.etal/4h2o.temelso.etal.dset.1h2o-dset.npz'),
                dataSet(f'{dset_dir_h2o}/2h2o/temelso.etal/4h2o.temelso.etal.dset.2h2o-dset.mb.npz'),
                dataSet(f'{dset_dir_h2o}/3h2o/temelso.etal/4h2o.temelso.etal.dset.3h2o-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_h2o}/1h2o/temelso.etal/5h2o.temelso.etal.dset.1h2o-dset.npz'),
                dataSet(f'{dset_dir_h2o}/2h2o/temelso.etal/5h2o.temelso.etal.dset.2h2o-dset.mb.npz'),
                dataSet(f'{dset_dir_h2o}/3h2o/temelso.etal/5h2o.temelso.etal.dset.3h2o-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_h2o}/1h2o/temelso.etal/6h2o.temelso.etal.dset.1h2o-dset.npz'),
                dataSet(f'{dset_dir_h2o}/2h2o/temelso.etal/6h2o.temelso.etal.dset.2h2o-dset.mb.npz'),
                dataSet(f'{dset_dir_h2o}/3h2o/temelso.etal/6h2o.temelso.etal.dset.3h2o-dset.mb.npz'),
            ],
        ],
        'mecn': [
            [
                dataSet(f'{dset_dir_mecn}/1mecn/malloum.etal/4mecn.malloum.etal.dset.1mecn-dset.npz'),
                dataSet(f'{dset_dir_mecn}/2mecn/malloum.etal/4mecn.malloum.etal.dset.2mecn-dset.mb.npz'),
                dataSet(f'{dset_dir_mecn}/3mecn/malloum.etal/4mecn.malloum.etal.dset.3mecn-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_mecn}/1mecn/malloum.etal/5mecn.malloum.etal.dset.1mecn-dset.npz'),
                dataSet(f'{dset_dir_mecn}/2mecn/malloum.etal/5mecn.malloum.etal.dset.2mecn-dset.mb.npz'),
                dataSet(f'{dset_dir_mecn}/3mecn/malloum.etal/5mecn.malloum.etal.dset.3mecn-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_mecn}/1mecn/malloum.etal/6mecn.malloum.etal.dset.1mecn-dset.npz'),
                dataSet(f'{dset_dir_mecn}/2mecn/malloum.etal/6mecn.malloum.etal.dset.2mecn-dset.mb.npz'),
                dataSet(f'{dset_dir_mecn}/3mecn/malloum.etal/6mecn.malloum.etal.dset.3mecn-dset.mb.npz'),
            ],
        ],
        'meoh': [
            [
                dataSet(f'{dset_dir_meoh}/1meoh/boyd.etal/4meoh.boyd.etal.dset.1meoh-dset.npz'),
                dataSet(f'{dset_dir_meoh}/2meoh/boyd.etal/4meoh.boyd.etal.dset.2meoh-dset.mb.npz'),
                dataSet(f'{dset_dir_meoh}/3meoh/boyd.etal/4meoh.boyd.etal.dset.3meoh-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_meoh}/1meoh/boyd.etal/5meoh.boyd.etal.dset.1meoh-dset.npz'),
                dataSet(f'{dset_dir_meoh}/2meoh/boyd.etal/5meoh.boyd.etal.dset.2meoh-dset.mb.npz'),
                dataSet(f'{dset_dir_meoh}/3meoh/boyd.etal/5meoh.boyd.etal.dset.3meoh-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_meoh}/1meoh/boyd.etal/6meoh.boyd.etal.dset.1meoh-dset.npz'),
                dataSet(f'{dset_dir_meoh}/2meoh/boyd.etal/6meoh.boyd.etal.dset.2meoh-dset.mb.npz'),
                dataSet(f'{dset_dir_meoh}/3meoh/boyd.etal/6meoh.boyd.etal.dset.3meoh-dset.mb.npz'),
            ],
        ]
    },
    'augccpvtz': {
        'h2o': [
            [
                dataSet(f'{dset_dir_h2o}/1h2o/temelso.etal/4h2o.temelso.etal.dset.1h2o-dset-mp2.augccpvtz.npz'),
                dataSet(f'{dset_dir_h2o}/2h2o/temelso.etal/4h2o.temelso.etal.dset.2h2o-dset.mb-mp2.augccpvtz.npz'),
                dataSet(f'{dset_dir_h2o}/3h2o/temelso.etal/4h2o.temelso.etal.dset.3h2o-dset.mb-mp2.augccpvtz.npz'),
            ],
            [
                dataSet(f'{dset_dir_h2o}/1h2o/temelso.etal/5h2o.temelso.etal.dset.1h2o-dset-mp2.augccpvtz.npz'),
                dataSet(f'{dset_dir_h2o}/2h2o/temelso.etal/5h2o.temelso.etal.dset.2h2o-dset.mb-mp2.augccpvtz.npz'),
                dataSet(f'{dset_dir_h2o}/3h2o/temelso.etal/5h2o.temelso.etal.dset.3h2o-dset.mb-mp2.augccpvtz.npz'),
            ],
            [
                dataSet(f'{dset_dir_h2o}/1h2o/temelso.etal/6h2o.temelso.etal.dset.1h2o-dset-mp2.augccpvtz.npz'),
                dataSet(f'{dset_dir_h2o}/2h2o/temelso.etal/6h2o.temelso.etal.dset.2h2o-dset.mb-mp2.augccpvtz.npz'),
                dataSet(f'{dset_dir_h2o}/3h2o/temelso.etal/6h2o.temelso.etal.dset.3h2o-dset.mb-mp2.augccpvtz.npz'),
            ],
        ],
        'mecn': [
            [
                dataSet(f'{dset_dir_mecn}/1mecn/malloum.etal/4mecn.malloum.etal.dset.1mecn-dset.npz'),
                dataSet(f'{dset_dir_mecn}/2mecn/malloum.etal/4mecn.malloum.etal.dset.2mecn-dset.mb.npz'),
                dataSet(f'{dset_dir_mecn}/3mecn/malloum.etal/4mecn.malloum.etal.dset.3mecn-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_mecn}/1mecn/malloum.etal/5mecn.malloum.etal.dset.1mecn-dset.npz'),
                dataSet(f'{dset_dir_mecn}/2mecn/malloum.etal/5mecn.malloum.etal.dset.2mecn-dset.mb.npz'),
                dataSet(f'{dset_dir_mecn}/3mecn/malloum.etal/5mecn.malloum.etal.dset.3mecn-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_mecn}/1mecn/malloum.etal/6mecn.malloum.etal.dset.1mecn-dset.npz'),
                dataSet(f'{dset_dir_mecn}/2mecn/malloum.etal/6mecn.malloum.etal.dset.2mecn-dset.mb.npz'),
                dataSet(f'{dset_dir_mecn}/3mecn/malloum.etal/6mecn.malloum.etal.dset.3mecn-dset.mb.npz'),
            ],
        ],
        'meoh': [
            [
                dataSet(f'{dset_dir_meoh}/1meoh/boyd.etal/4meoh.boyd.etal.dset.1meoh-dset.npz'),
                dataSet(f'{dset_dir_meoh}/2meoh/boyd.etal/4meoh.boyd.etal.dset.2meoh-dset.mb.npz'),
                dataSet(f'{dset_dir_meoh}/3meoh/boyd.etal/4meoh.boyd.etal.dset.3meoh-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_meoh}/1meoh/boyd.etal/5meoh.boyd.etal.dset.1meoh-dset.npz'),
                dataSet(f'{dset_dir_meoh}/2meoh/boyd.etal/5meoh.boyd.etal.dset.2meoh-dset.mb.npz'),
                dataSet(f'{dset_dir_meoh}/3meoh/boyd.etal/5meoh.boyd.etal.dset.3meoh-dset.mb.npz'),
            ],
            [
                dataSet(f'{dset_dir_meoh}/1meoh/boyd.etal/6meoh.boyd.etal.dset.1meoh-dset.npz'),
                dataSet(f'{dset_dir_meoh}/2meoh/boyd.etal/6meoh.boyd.etal.dset.2meoh-dset.mb.npz'),
                dataSet(f'{dset_dir_meoh}/3meoh/boyd.etal/6meoh.boyd.etal.dset.3meoh-dset.mb.npz'),
            ],
        ]
    }
}

save_dir = '../../analysis/isomer-predictions'

if save_dir[-1] != '/':
    save_dir += '/'

mbgdml_colors = {
    'h2o': '#4ABBF3',
    'mecn': '#61BFA3',
    'meoh': '#FFB5BA',
}
mbe_colors = {
    'h2o': '#b6e3fa',
    'mecn': '#bfe5da',
    'meoh': '#ffe1e3',
}
ref_color = 'silver'


# FIGURE #

# Setting up general figure properties
font = {'family' : 'sans-serif',
        'size'   : 8}
plt.rc('font', **font)

include_ref_values = False

def dset_mp2(nmer_dset, E_idx=None, E_ref=None):
    mp2_E = nmer_dset.E
    if E_idx is None:
        E_idx = np.argsort(mp2_E)
    mp2_E = mp2_E[E_idx]
    if E_ref is None:
        E_ref = mp2_E[0]
    # E_relative = mp2_E - E_ref
    return E_idx, mp2_E

def dset_mbe(nmer_dset, all_mbe_dsets, E_idx):
    E_ref = nmer_dset.E[E_idx][0]

    mb_dset = copy(nmer_dset)
    mb_dset.E = np.zeros(mb_dset.E.shape)
    mb_dset = e_f_contribution(mb_dset, all_mbe_dsets, 'add')
    mbe_E = mb_dset.E[E_idx]
    # mbe_E_relative = mbe_E - E_ref
    return mbe_E

for solvent in solvents:
    fig, axes = plt.subplots(1, 3 , figsize=(6, 3.5), constrained_layout=True)

    data_color = mbgdml_colors[solvent]
    mbe_color = mbe_colors[solvent]
    line_width = 1.5
    marker_style = 'o'
    marker_size = 5

    for i in range(len(isomer_dsets['def2tzvp'][solvent])):
        mp2_E_relative = {}
        mbe_E_relative = {}

        # Relative energies of all data are with respect to MP2 lowest.
        E_idx_aug, mp2_E_relative['augccpvtz'] = dset_mp2(
            isomer_dsets['augccpvtz'][solvent][i], E_idx=None, E_ref=None
        )
        E_mp2_aug_ref = mp2_E_relative['augccpvtz'][0]

        E_idx_def2, mp2_E_relative['def2tzvp'] = dset_mp2(
            isomer_dsets['def2tzvp'][solvent][i], E_idx=None, E_ref=None
        )
        E_mp2_def2_ref = mp2_E_relative['def2tzvp'][0]

        # many-body expansion
        all_mbe_dsets_aug = isomer_mb_dsets['augccpvtz'][solvent][i]
        all_mbe_dsets_def2 = isomer_mb_dsets['def2tzvp'][solvent][i]
        mbe_E_relative['augccpvtz'] = dset_mbe(
            isomer_dsets['augccpvtz'][solvent][i],
            all_mbe_dsets_aug, E_idx_aug
        )
        mbe_E_relative['def2tzvp'] = dset_mbe(
            isomer_dsets['def2tzvp'][solvent][i],
            all_mbe_dsets_def2, E_idx_def2
        )
        
        # Prints model performance information.
        aug_mae = np.mean(np.abs(mbe_E_relative['augccpvtz'] - mp2_E_relative['augccpvtz']))
        def2_mae = np.mean(np.abs(mbe_E_relative['def2tzvp'] - mp2_E_relative['def2tzvp']))
        print(f'{solvent} {i+4}mer MAE aug: {aug_mae:.4f} kcal/mol    def2: {def2_mae:.4f} kcal/mol\n')

