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

"""Starts NVT simulation from trajectory using initialized velocities."""

from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mbgdml.mbe import mbePredict
from mbgdml.models import gdmlModel
from mbgdml.predictors import predict_gdml
from mbgdml.descriptors import Criteria, com_distance_sum
from mbgdml.periodic import Cell
from mbgdml.interfaces.ase import mbeCalculator
from mbgdml.utils import get_entity_ids, get_comp_ids
import numpy as np
import os

# System information
work_dir = '/ihome/jkeith/amm503/projects/mbgdml-h2o-meoh-mecn/data/md/h2o/58h2o/58h2o-mbgdml-nvt-init_100-tau_0.01-1'

# Structure information
traj_start_path = '/ihome/jkeith/amm503/projects/mbgdml-h2o-meoh-mecn/data/md/h2o/58h2o/58h2o-mbgdml-nvt-init_100-tau_0.01-1/58h2o-mbgdml-opt.traj'
traj_idx = -1  # Index of the structure to start MD simulation with.

n_molecules = 58
entity_ids = get_entity_ids(3, n_molecules)
comp_ids = get_comp_ids('h2o', n_molecules, entity_ids)

box_length = 12.0  # Angstroms
pbc_cutoff = 6.0  # Maximum value is half of the box length
cell_v = [[box_length, 0.0, 0.0],
          [0.0, box_length, 0.0],
          [0.0, 0.0, box_length]]
periodic_cell = Cell(cell_v, pbc=True, cutoff=pbc_cutoff)

# NVT parameters
traj_name = '58h2o-mbgdml-nvt-init_100-tau_0.01-1'
traj_interval = 1  # Log trajectory every _ steps
t_step = 1.0  # Integration time step in fs
n_steps = 10000  # Number of MD steps
init_temp = 100  # K
temp = 298.15  # K
taut = 0.01  # ps

# Model information
model_paths = [
    'h2o/1h2o/gdml/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-train1000.npz',
    'h2o/2h2o/gdml/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6-model.mb-train1000.npz',
    'h2o/3h2o/gdml/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-train1000.npz',
]
mb_cutoffs = [None, 6.0, 10.0]
e_units = 'kcal/mol'  # kcal/mol, eV

# Parallel information
use_ray = True
n_workers = 12



###   SCRIPT   ###

# Sets up paths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
model_dir = '/ihome/jkeith/amm503/projects/mbgdml-h2o-meoh-mecn/models'

model_paths = [
    os.path.join(model_dir, model_path) for model_path in model_paths
]

# Prepares models
model_desc_kwargs = (
    {'entity_ids': get_entity_ids(atoms_per_mol=3, num_mol=1)},  # 1h2o
    {'entity_ids': get_entity_ids(atoms_per_mol=3, num_mol=2)},  # 2h2o 
    {'entity_ids': get_entity_ids(atoms_per_mol=3, num_mol=3)},  # 3h2o
)
model_desc_cutoffs = (None, 6.0, 10.0)
model_criterias = [
    Criteria(com_distance_sum, desc_kwargs, cutoff) for desc_kwargs,cutoff \
    in zip(model_desc_kwargs, model_desc_cutoffs)
]
models = []
for i in range(len(model_paths)):
    models.append(
        gdmlModel(
            model_paths[i], criteria=model_criterias[i]
        )
    )

# Conversion factors
hartree2kcalmol = 627.5094737775374055927342256  # Psi4 constant
hartree2ev = 27.21138602  # Psi4 constant
hartree2kJmol = 2625.4996382852165050  # Psi4 constant
ev2kcalmol = hartree2kcalmol/hartree2ev
kcalmol2ev = hartree2ev/hartree2kcalmol

if e_units.lower() == 'kcal/mol':
    e_conv = kcalmol2ev
    f_conv = kcalmol2ev
elif e_units.lower() == 'ev':
    e_conv = 1.0
    f_conv = 1.0


mbe_pred = mbePredict(
    models, predict_gdml, use_ray=use_ray, n_workers=n_workers,
    periodic_cell=periodic_cell
)


# Setup ase system
traj_start = Trajectory(traj_start_path)
ase_atoms = traj_start[traj_idx]

# Attach ASE calculator
mbe_calc = mbeCalculator(mbe_pred, e_conv=e_conv, f_conv=f_conv)
mbe_calc.directory = work_dir
mbe_calc.set(entity_ids=entity_ids, comp_ids=comp_ids)
ase_atoms.calc = mbe_calc

# Setup trajectory
md_traj_path = os.path.join(work_dir, traj_name+'.traj')
traj = Trajectory(
    md_traj_path, mode='w', atoms=ase_atoms
)

# Setup logging to monitor progress
i_print_step = 0
def print_log(a=ase_atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    global i_print_step, n_steps
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    ekin_per_atom = ekin / len(a)
    temp = ekin_per_atom / (1.5 * units.kB)
    print(
        f'Step {i_print_step}/{n_steps}: E_pot = {epot:.8f} eV  '\
        f'E_kin = {ekin:.8f} eV (T = {temp:.2f} K)  '\
        f'Etot = {epot+ekin:.8f} eV'
    )
    i_print_step += 1

# Initialize velocities
MaxwellBoltzmannDistribution(ase_atoms, temperature_K=init_temp)

# Setup NVT
md = NVTBerendsen(
    atoms=ase_atoms, timestep=t_step*units.fs, temperature_K=temp, fixcm=True,
    taut=taut*1000*units.fs
)
md.attach(print_log, interval=1)
md.attach(traj.write, interval=traj_interval)


# Run
md.run(n_steps)
