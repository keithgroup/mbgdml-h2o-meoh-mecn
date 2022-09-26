import os
import schnetpack as spk
import sys
import numpy as np

db_path = 'meoh/62meoh.sphere.gfn2.md.500k.prod1-dset-all.db'
save_dir = 'meoh/'


##################
###   SCRIPT   ###
##################

# Setup paths
job_dir = '/ihome/jkeith/amm503/projects/schnet-training'

db_path = os.path.join(job_dir, db_path)
save_dir = os.path.join(job_dir, save_dir)


# Load data
molecule_data = spk.AtomsData(db_path) 
atoms, properties = molecule_data.get_properties(0)


print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])

print('Forces:\n', properties['forces'])
print('Shape:\n', properties['forces'].shape)


# Split dataset
train, val, test = spk.data.train_test_split(
    data=molecule_data,
    num_train=3000,
    num_val=6000,
    split_file=os.path.join(save_dir, 'meoh-schnet-split.npz')
)

train_loader = spk.AtomsLoader(train, shuffle=True) #batch_size=40, shuffle=True)
val_loader = spk.AtomsLoader(val) #, batch_size=10)

means, stddevs = train_loader.get_statistics(
    'energy', divide_by_atoms=True
)

print('Mean atomization energy / atom:      {:12.6f} [eV]'.format(means['energy'][0]))
print('Std. dev. atomization energy / atom: {:12.6f} [eV]'.format(stddevs['energy'][0]))

n_features = 128
n_interactions = 5
cutoff = 10.
n_gaussians = 100
schnet = spk.representation.SchNet(
    n_atom_basis=n_features,
    n_filters=n_features,
    n_gaussians=n_gaussians,
    n_interactions=n_interactions,
    cutoff=cutoff,
    cutoff_network=spk.nn.cutoff.CosineCutoff
)
print(f'n_features: {n_features}')
print(f'n_interactions: {n_interactions}')
print(f'cutoff: {cutoff}')
print(f'n_gaussians: {n_gaussians}')

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
    trn.CSVHook(log_path=save_dir, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path=save_dir,
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

print(f'Start training using {device}')
trainer.train(device=device)

###   TEST PREDICTIONS   ###

best_model = torch.load(os.path.join(save_dir, 'best_model'))

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

energy_error /= len(test)
forces_error /= len(test)

print('\nTest MAE:')
print('    energy: {:10.6f} eV'.format(energy_error))
print('    forces: {:10.6f} eV/\u212B'.format(forces_error))

