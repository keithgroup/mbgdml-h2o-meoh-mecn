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

"""Plots force MAEs of models trained with increasing more data (in one plot)."""

import numpy as np
import matplotlib.pyplot as plt
from mbgdml.data import predictSet
import pickle

solvents = ['h2o', 'mecn', 'meoh']  # 'h2o', 'mecn', 'meoh'




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Plot names
plot_names = {
    'h2o': 'h2o-mbgdml-training-curve',
    'mecn': 'mecn-mbgdml-training-curve',
    'meoh': 'meoh-mbgdml-training-curve',
}

# Iterative training paths.
iter_train_dir = '../../analysis/iterative-training'
iter_train_dir_h2o = f'{iter_train_dir}/h2o'
iter_train_dir_mecn = f'{iter_train_dir}/mecn'
iter_train_dir_meoh = f'{iter_train_dir}/meoh'

iterative_training_pickles = {
    'h2o': [
        (
            pickle.load(open(f'{iter_train_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-randomtrain200/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain300/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain400/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain500/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain600/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain700/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain800/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain900/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain1000/saves/cluster_error_140h2o/info.p', 'rb')),
        ),
        (
            pickle.load(open(f'{iter_train_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.mb.cm6-model-randomtrain200/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.mb.cm6-model-iterativetrain300/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.mb.cm6-model-iterativetrain400/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.mb.cm6-model-iterativetrain500/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.mb.cm6-model-iterativetrain600/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.mb.cm6-model-iterativetrain700/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.mb.cm6-model-iterativetrain800/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.mb.cm6-model-iterativetrain900/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.mb.cm6-model-iterativetrain1000/saves/cluster_error_140h2o/info.p', 'rb')),
        ),
        (
            pickle.load(open(f'{iter_train_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-randomtrain200/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain300/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain400/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain500/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain600/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain700/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain800/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain900/saves/cluster_error_140h2o/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_h2o}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain1000/saves/cluster_error_140h2o/info.p', 'rb')),
        ),
    ],
    'mecn': [
        (
            pickle.load(open(f'{iter_train_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-randomtrain200/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain300/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain400/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain500/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain600/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain700/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain800/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain900/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/1mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.1mecn-model-iterativetrain1000/saves/cluster_error_48mecn/info.p', 'rb')),
        ),
        (
            pickle.load(open(f'{iter_train_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-randomtrain200/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain300/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain400/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain500/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain600/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain700/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain800/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain900/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/2mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17.dset.2mecn.cm9-model.mb-iterativetrain1000/saves/cluster_error_48mecn/info.p', 'rb')),
        ),
        (
            pickle.load(open(f'{iter_train_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-randomtrain200/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain300/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain400/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain500/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain600/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain700/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain800/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain900/saves/cluster_error_48mecn/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_mecn}/3mecn/48mecn.sphere.gfn2.md.500k.prod1.3mecn.cm17-model.mb-iterativetrain1000/saves/cluster_error_48mecn/info.p', 'rb')),
        ),
    ],
    'meoh': [
        (
            pickle.load(open(f'{iter_train_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-randomtrain200/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-iterativetrain300/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-iterativetrain400/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-iterativetrain500/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-iterativetrain600/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-iterativetrain700/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-iterativetrain800/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-iterativetrain900/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/1meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.1meoh-model-iterativetrain1000/saves/cluster_error_62meoh/info.p', 'rb')),
        ),
        (
            pickle.load(open(f'{iter_train_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-randomtrain200/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain300/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain400/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain500/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain600/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain700/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain800/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain900/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain1000/saves/cluster_error_62meoh/info.p', 'rb')),
        ),
        (
            pickle.load(open(f'{iter_train_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-randomtrain200/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain300/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain400/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain500/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain600/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain700/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain800/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain900/saves/cluster_error_62meoh/info.p', 'rb')),
            pickle.load(open(f'{iter_train_dir_meoh}/3meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14-model.mb-iterativetrain1000/saves/cluster_error_62meoh/info.p', 'rb')),
        ),
    ],
}

training_sizes = np.array([i*100 for i in range(2, 11)])

# pset paths.
pset_dir = '../../data/predictsets'
pset_dir_h2o = f'{pset_dir}/h2o'
pset_dir_mecn = f'{pset_dir}/mecn'
pset_dir_meoh = f'{pset_dir}/meoh'

pset_paths = {
    'h2o': [
        predictSet(f'{pset_dir_h2o}/4h2o/4h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_h2o}/5h2o/5h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_h2o}/6h2o/6h2o.temelso.etal-pset-140h2o.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz')
    ],
    'mecn': [
        predictSet(f'{pset_dir_mecn}/4mecn/4mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_mecn}/5mecn/5mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_mecn}/6mecn/6mecn.malloum.etal-pset-48mecn.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz')
    ],
    'meoh': [
        predictSet(f'{pset_dir_meoh}/4meoh/4meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_meoh}/5meoh/5meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz'),
        predictSet(f'{pset_dir_meoh}/6meoh/6meoh.boyd.etal-pset-62meoh.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz')
    ]
}

save_dir = '../../analysis/iterative-training'

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



##################
###   FIGURE   ###
##################

# Setting up general figure properties
font = {'family' : 'sans-serif',
        'size'   : 8}
plt.rc('font', **font)

include_ref_values = False

line_styles = ['-', 'dashed', (0, (5, 10))]
markers = ['o', 's', '^']

for solvent in solvents:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), constrained_layout=True)

    color = mbgdml_colors[solvent]

    solvent_training_pickles = iterative_training_pickles[solvent]
    
    for i_nbody in range(len(solvent_training_pickles)):
        all_nbody_pickles = solvent_training_pickles[i_nbody]
        training_mae = np.zeros(len(all_nbody_pickles))
        for i_training_size in range(len(all_nbody_pickles)):
            nbody_pickle = all_nbody_pickles[i_training_size]
            training_mae[i_training_size] = nbody_pickle['errors']['MAE_o']  # kcal/(mol A)
        

        ax.plot(
            training_sizes, training_mae,
            marker=markers[i_nbody], markersize=5,
            linestyle=line_styles[i_nbody], linewidth=1.5,
            color=color, label=f'{i_nbody+1}-body'
        )
    
    ax.set_xlabel('Training Set Size')

    ax.set_ylabel('Force MAE (kcal/(mol $\AA$))')

    ax.legend(frameon=False)

    plt_path = f'{save_dir}{plot_names[solvent]}.png'

    print(f'Saving {plot_names[solvent]}')
    plt.savefig(plt_path, dpi=1000)

