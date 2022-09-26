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

"""Trains a SchNet potential."""

import os
import schnetpack as spk
import sys
import numpy as np

db_path = 'h2o/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-dset.db'
model_path = 'h2o/1h2o/model/'

use_gdml_idxs = True
mbgdml_model_path = 'h2o/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain1000.npz'


##################
###   SCRIPT   ###
##################

# Setup paths
job_dir = '../../../'

db_path = os.path.join(job_dir, db_path)
model_path = os.path.join(job_dir, model_path)
mbgdml_model_path = os.path.join(job_dir, mbgdml_model_path)
if not os.path.exists(model_path):
	os.makedirs(model_path)

# Load data
molecule_data = spk.AtomsData(db_path) 
atoms, properties = molecule_data.get_properties(0)


print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])

print('Forces:\n', properties['forces'])
print('Shape:\n', properties['forces'].shape)


# Determine subsets
mbgdml_model = dict(np.load(mbgdml_model_path, allow_pickle=True))
mbgdml_idxs_train = mbgdml_model['idxs_train']
mbgdml_idxs_val = mbgdml_model['idxs_valid']
mbgdml_idxs_test = np.full(len(molecule_data), True)
mbgdml_idxs_test[mbgdml_idxs_train] = False
mbgdml_idxs_test[mbgdml_idxs_val] = False
mbgdml_idxs_test = np.argwhere(mbgdml_idxs_test).flatten()
# Convert all to Python int
mbgdml_idxs_train = [int(i) for i in mbgdml_idxs_train]
mbgdml_idxs_val = [int(i) for i in mbgdml_idxs_val]
mbgdml_idxs_test = [int(i) for i in mbgdml_idxs_test]

train = spk.AtomsDataSubset(molecule_data, mbgdml_idxs_train)
val = spk.AtomsDataSubset(molecule_data, mbgdml_idxs_val)
test = spk.AtomsDataSubset(molecule_data, mbgdml_idxs_test)

train_loader = spk.AtomsLoader(train, shuffle=True) #batch_size=40, shuffle=True)
val_loader = spk.AtomsLoader(val) #, batch_size=10)

means, stddevs = train_loader.get_statistics(
    'energy', divide_by_atoms=True
)

print('Mean atomization energy / atom:      {:12.6f} [kcal/mol]'.format(means['energy'][0]))
print('Std. dev. atomization energy / atom: {:12.6f} [kcal/mol]'.format(stddevs['energy'][0]))

n_features = 128

schnet = spk.representation.SchNet(
    n_atom_basis=n_features,
    n_filters=n_features,
    n_gaussians=25,
    n_interactions=5,
    cutoff=5.,
    cutoff_network=spk.nn.cutoff.CosineCutoff
)

energy_model = spk.atomistic.Atomwise(
    n_in=n_features,
    property='energy',
    mean=means['energy'],
    stddev=stddevs['energy'],
    derivative='forces',
    negative_dr=True
)

model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)

import torch

# tradeoff
rho_tradeoff = 0.1

# loss function
def loss(batch, result):
	# compute the mean squared error on the energies
	diff_energy = batch['energy']-result['energy']
	err_sq_energy = torch.mean(diff_energy ** 2)

	# compute the mean squared error on the forces
	diff_forces = batch['forces']-result['forces']
	err_sq_forces = torch.mean(diff_forces ** 2)

	# build the combined loss function
	err_sq = rho_tradeoff*err_sq_energy + (1-rho_tradeoff)*err_sq_forces

	return err_sq

from torch.optim import Adam

# build optimizer
optimizer = Adam(model.parameters(), lr=5e-4)


import schnetpack.train as trn

# set up metrics
metrics = [
    spk.metrics.MeanAbsoluteError('energy'),
    spk.metrics.MeanAbsoluteError('forces')
]

# construct hooks
hooks = [
    trn.CSVHook(log_path=model_path, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path=model_path,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

trainer.train(device=device)

###   TEST PREDICTIONS   ###

best_model = torch.load(os.path.join(model_path, 'best_model'))

test_loader = spk.AtomsLoader(test)#, batch_size=100)

energy_error = 0.0
forces_error = 0.0

for count, batch in enumerate(test_loader):
	# move batch to GPU, if necessary
	batch = {k: v.to(device) for k, v in batch.items()}

	# apply model
	pred = best_model(batch)

	# calculate absolute error of energies
	tmp_energy = torch.sum(torch.abs(pred['energy'] - batch['energy']))
	tmp_energy = tmp_energy.detach().cpu().numpy() # detach from graph & convert to numpy
	energy_error += tmp_energy

	# calculate absolute error of forces, where we compute the mean over the n_atoms x 3 dimensions
	tmp_forces = torch.sum(
    	torch.mean(torch.abs(pred['forces'] - batch['forces']), dim=(1,2))
	)
	tmp_forces = tmp_forces.detach().cpu().numpy() # detach from graph & convert to numpy
	forces_error += tmp_forces

	# log progress
	percent = '{:3.2f}'.format(count/len(test_loader)*100)
	print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")

energy_error /= len(test)
forces_error /= len(test)

print('\nTest MAE:')
print('    energy: {:10.6f} kcal/mol'.format(energy_error))
print('    forces: {:10.6f} kcal/mol/\u212B'.format(forces_error))


