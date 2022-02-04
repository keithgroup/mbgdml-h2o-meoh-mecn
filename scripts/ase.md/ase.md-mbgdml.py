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

"""Prepares and runs a MD simulation with ASE"""

import os
import numpy as np

from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

from cclib.parser.utils import convertor

from mbgdml.utils import get_entity_ids, get_comp_ids, convert_forces, write_xyz
from mbgdml.md import mbGDML_ASE_Calculator

###   SCRIPT PARAMETERS   ###

model_dir = '../../data/models'
model_dir_h2o = f'{model_dir}/h2o'
model_dir_mecn = f'{model_dir}/mecn'
model_dir_meoh = f'{model_dir}/meoh'

model_dir_solvent = model_dir_h2o

model_paths = [
    f'{model_dir_solvent}/1h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.1h2o-model-iterativetrain500.npz',
    f'{model_dir_solvent}/2h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10.dset.2h2o.cm6-model.mb-iterativetrain500.npz',
    f'{model_dir_solvent}/3h2o/140h2o.sphere.gfn2.md.500k.prod1.3h2o.cm10-model.mb-iterativetrain500.npz',
]

structure_path = './6h2o.temelso.etal.pr.xyz'


solvent = 'h2o'
num_solvent_molecules = 6
t_step = 1  # fs
steps = 10  # Number of time steps
temp = 300  # K




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)


# Prepares entity_ids and comp_ids

if solvent == 'h2o':
    atoms_per_mol = 3
elif solvent == 'mecn' or solvent == 'meoh':
    atoms_per_mol = 6
entity_ids = get_entity_ids(atoms_per_mol, num_solvent_molecules)
comp_ids = get_comp_ids(solvent, entity_ids)



def store_data(i, ase_structure):
    R[i] = ase_structure.arrays['positions']
    E_potential[i] = ase_structure.get_potential_energy()
    E_kinetic[i] = ase_structure.get_kinetic_energy()
    E_total[i] = E_potential[i] + E_kinetic[i]
    F[i] = ase_structure.get_forces()
    e_kin_per_atom = E_kinetic[i] / len(ase_atoms)
    temps[i] = e_kin_per_atom / (1.5 * units.kB)
    Vel[i] = ase_structure.get_velocities()

def print_data(step_i, total_steps, e_pot, e_kin, temp):
    """Quick function to print MD information during simulation.
    
    Parameters
    ----------
    a : :obj:`ase.atoms`
        Atoms object from ASE.
    """
    print(
        f'Step {step_i}/{total_steps}: E_pot = {e_pot:.8f} eV  '\
        f'E_kin = {e_kin:.8f} eV (T = {temp:.2f} K)  '\
        f'Etot = {e_pot+e_kin:.8f} eV'
    )






###   SETTING UP MD SIMULATION   ###

structure_name = os.path.splitext(os.path.basename(structure_path))[0]

# Initalizing structure.
ase_structure = read(structure_path)
ase_atoms = ase_structure.numbers

# Initalizing calculator.
mbgdml_calculator = mbGDML_ASE_Calculator(
    ase_atoms, model_paths, entity_ids, comp_ids
)
mbgdml_calculator.directory = working_dir
ase_structure.set_calculator(mbgdml_calculator)

# Initalizing velocities according to temperature.
MaxwellBoltzmannDistribution(ase_structure, temperature_K=temp)

# print(ase_structure.__dict__)

# Setting up MD with constant energy using the VelocityVerlet algorithm
md_name = f'{structure_name}-{steps}fs-{temp}K'
traj = Trajectory(f'{md_name}.traj', mode='w', atoms=ase_structure)
dyn = VelocityVerlet(
    ase_structure,
    timestep=t_step * units.fs,
    append_trajectory=False,
    logfile=f'{md_name}.log',
    loginterval=1
)
dyn.attach(traj.write, interval=1)

# Setting up variables to capture MD data.
# In ASE, momentum is eV^(1/2) amu^(1/2) so velocities are eV^(1/2) / amu^(1/2)
r_shape = ase_structure.arrays['positions'].shape
R = np.zeros((steps + 1, r_shape[0], r_shape[1]))  # Ang
E_potential = np.zeros((steps + 1))  # eV
E_kinetic = np.zeros((steps + 1))  # eV
E_total = np.zeros((steps + 1))  # eV
F = np.zeros((steps + 1, r_shape[0], r_shape[1]))  # eV/A
Vel = np.zeros((steps + 1, r_shape[0], r_shape[1]))  # 1 sqrt(eV / amu) = 0.0982269475 Ang / fs
temps = np.zeros((steps + 1))  # K

# Data from starting structure.
store_data(0, ase_structure)
print_data(0, steps, E_potential[0], E_kinetic[0], temps[0])



###   RUNNING MD SIMULATION   ###
md_name = f'{structure_name}-{steps}fs-{temp}K'

# Running MD simulation.
for i in range(1, steps + 1):
    dyn.run(1)
    # Get data from step
    store_data(i, ase_structure)
    print_data(i, steps, E_potential[i], E_kinetic[i], temps[i])


###   POST MD PROCESSING   ###

# Converting units of energy from eV to kcal/mol
E_potential = convertor(E_potential, 'eV', 'kcal/mol')
E_kinetic = convertor(E_kinetic, 'eV', 'kcal/mol')
E_total = convertor(E_total, 'eV', 'kcal/mol')
F = convert_forces(F, 'eV', 'Angstrom', 'kcal/mol', 'Angstrom')

# Converting velocities from sqrt(eV / amu) to Ang / fs
Vel *= 0.0982269475

# Creating MD results npz.
md_data = {
    'z': ase_atoms,
    'R': R,
    'E_potential': E_potential,
    'E_kinetic': E_kinetic,
    'F': F,
    'V': Vel,
    'type': 'md',
    'e_unit': 'kcal/mol',
    'r_unit': 'Angstrom',
    'v_unit': 'Angstrom/fs',
    'entity_ids': entity_ids,
    'comp_ids': comp_ids
}

save_path = working_dir + md_name + '.npz'
np.savez_compressed(save_path, **md_data)
write_xyz(ase_atoms, R, working_dir, md_name)
