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

import os
import math
import subprocess

###   CHANGE   ###

structure_dir = '.'
monomer_path = f'{structure_dir}/1h2o.abc.mp2.def2tzvp.xyz'
save_dir = '.'

solvent_label = 'h2o'
mass_density = 997.776  # kg/m3 @ desired temperature and pressure
molar_mass = 18.015  # g/mol

diameter = 13  # Angstroms

###   INFORMATION   ###

# Liquid densities were found using
# http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe
# h2o @ 300 K: 997.776 kg/m3
# mecn @ 300 K: 776.894 kg/m3
# meoh @ 300 K: 786.005 kg/m3

# Molar masses are from pubchem.
# h2o: 18.015 g/mol
# mecn: 41.05 g/mol
# meoh: 32.042 g/mol




###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

NA = 6.02214076e23  # particles/mole

if save_dir[-1] != '/':
    save_dir += '/'

# Substance properties.
molecular_mass = molar_mass / 1000  # convert to kg/mol
mole_density = mass_density / molecular_mass  # mol/m3

# Sphere properties.
radius = diameter/2  # diameter to radius in Angstroms
volume = (4/3) * math.pi * (radius * 1e-10)**3  # Volume of sphere in m3

# Number of atoms in the sphere.
moles = mole_density * volume
molecules = round(moles * NA)

# Checking that we want to move forward with packmol.
print(f'Liquid density: {mass_density} kg/m3')
print(f'Molar mass: {molecular_mass*1000} g/mol')
print(f'Sphere diameter: {diameter} Angstroms\n')
print(f'This sphere would need {molecules} molecules\n')

val = input("Would you like to proceed with packmol? (y/n): ")

if val == 'n':
    print('Stopping')
    exit()
elif val != 'y':
    print(f'"{val}" is not y or n')
    print('Stopping')
    exit()


# Packmol

os.chdir(save_dir)

packmol_input = [
    f'tolerance 2.0',
    f'output {molecules}{solvent_label}.sphere-packmol.xyz',
    f'filetype xyz',
    f'structure {monomer_path}',
    f'    number {molecules}',
    f'    inside sphere 0.0 0.0 0.0 {float(radius)}',
    f'end structure'
]
packmol_input = [i + '\n' for i in packmol_input]

packmol_input_name = f'{molecules}{solvent_label}.sphere-packmol.in'
packmol_input_path = f'{save_dir}{packmol_input_name}'
with open(packmol_input_path, 'w') as f:
    f.writelines(packmol_input)

# Executes Packmol.

completed_process = subprocess.run(
    f'packmol < {packmol_input_name}',
    capture_output=True,
    shell=True
)

packmol_output_name = packmol_input_name[:-2] + 'out'
packmol_output_path = f'{save_dir}{packmol_output_name}'
with open(packmol_output_path, 'w') as f:
    f.write(completed_process.stdout.decode())

print('\nDone!')
