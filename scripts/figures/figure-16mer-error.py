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

import mbgdml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mbgdml.data import predictSet, dataSet
from mbgdml.utils import e_f_contribution

#solvents = ['h2o', 'mecn', 'meoh']
solvents = ['h2o', 'mecn', 'meoh']
max_nbody_order = [3]

# Data set paths.
dset_dir = '../../data/datasets'
dset_dir_h2o = f'{dset_dir}/h2o'
dset_dir_mecn = f'{dset_dir}/mecn'
dset_dir_meoh = f'{dset_dir}/meoh'

# Predict set paths.
pset_dir = '../../data/predictsets'
pset_dir_h2o = f'{pset_dir}/h2o'
pset_dir_mecn = f'{pset_dir}/mecn'
pset_dir_meoh = f'{pset_dir}/meoh'

ref_16mer_dset_paths = {
    'h2o': f'{dset_dir_h2o}/16h2o/16h2o.yoo.etal.boat.b-dset-rimp2.def2tzvp.npz',
    'mecn': f'{dset_dir_mecn}/16mecn/16mecn.remya.etal-dset-rimp2.def2tzvp.npz',
    'meoh': f'{dset_dir_meoh}/16meoh/16meoh.pires.deturi-dset-rimp2.def2tzvp.npz',
}

ref_16mer_mbe_dset_paths = {
    'h2o': [
        f'{dset_dir_h2o}/1h2o/16h2o.yoo.etal.boat.b.1h2o-dset.npz',
        f'{dset_dir_h2o}/2h2o/16h2o.yoo.etal.boat.b.2h2o-dset.mb.npz',
        f'{dset_dir_h2o}/3h2o/16h2o.yoo.etal.boat.b.3h2o-dset.mb.npz',
    ],
    'mecn': [
        f'{dset_dir_mecn}/1mecn/16mecn.remya.etal.1mecn-dset.npz',
        f'{dset_dir_mecn}/2mecn/16mecn.remya.etal.2mecn-dset.mb.npz',
        f'{dset_dir_mecn}/3mecn/16mecn.remya.etal.3mecn-dset.mb.npz',
    ],
    'meoh': [
        f'{dset_dir_meoh}/1meoh/16meoh.pires.deturi.1meoh-dset.npz',
        f'{dset_dir_meoh}/2meoh/16meoh.pires.deturi.2meoh-dset.mb.npz',
        f'{dset_dir_meoh}/3meoh/16meoh.pires.deturi.3meoh-dset.mb.npz',
    ],
}

pset_16mers = {
    'h2o': f'{pset_dir_h2o}/16h2o/16h2o.yoo.etal.boat.b.rimp2-pset-140h2o.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz',
    'mecn': f'{pset_dir_mecn}/16mecn/16mecn.remya.etal.rimp2-pset-48mecn.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz',
    'meoh': f'{pset_dir_meoh}/16meoh/16meoh.pires.deturi.rimp2-pset-62meoh.sphere.gfn2.md.500k.prod1.iterativetrain1000.npz',
}

save_dir = '../../analysis/16mer'
plot_name = '16mer-error-iterativetrain1000'

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

# Setting up general figure properties
font = {'family' : 'sans-serif',
        'size'   : 8}
plt.rc('font', **font)

fig, ax1 = plt.subplots(1, 1, figsize=(3.25, 3.25), constrained_layout=True)
ax2 = ax1.twinx()

line_width = 1.5
marker_style = 'o'
marker_size = 4

E_error = []
#F_rmse = []
F_mae = []
xaxis_labels = []
E_colors = []

for solv in solvents:
    if solv == 'h2o':
        solv_label = 'H2O'
    elif solv == 'mecn':
        solv_label = 'MeCN'
    elif solv == 'meoh':
        solv_label = 'MeOH'

    ref_dset = dataSet(ref_16mer_dset_paths[solv])
    E_ref = ref_dset.E[0]
    F_ref = ref_dset.F

    for nbody_order in max_nbody_order:
        mbe_dsets = [dataSet(i) for i in ref_16mer_mbe_dset_paths[solv]]
        mbe_pred_dset = dataSet(ref_16mer_dset_paths[solv])
        mbe_pred_dset.E = np.zeros(mbe_pred_dset.E.shape)
        mbe_pred_dset.F = np.zeros(mbe_pred_dset.F.shape)
        mbe_pred_dset = e_f_contribution(mbe_pred_dset, mbe_dsets, 'add')
        E_mbe_pred = mbe_pred_dset.E[0]
        E_mbe_error = E_mbe_pred - E_ref

        F_mbe_pred = mbe_pred_dset.F[0]
        F_mbe_error = F_mbe_pred - F_ref
        #F_mbe_rmse = np.sqrt(np.mean((F_mbe_error)**2))
        #F_rmse.append(F_mbe_rmse)
        F_mbe_mae = np.mean(np.abs(F_mbe_error))
        F_mae.append(F_mbe_mae)

        E_error.append(E_mbe_error)
        xaxis_labels.append(f'{solv_label}\nMBE')
        E_colors.append(solvent_colors[solv])
        print(f'{solv_label} MBE Energy Error:    {E_mbe_error:.1f} kcal/mol')
        #print(f'{solv_label} MBE force RMSE with MP2: {F_mbe_rmse:.3f} kcal/mol/A')
        print(f'{solv_label} MBE force MAE with MP2: {F_mbe_mae:.3f} kcal/mol/A')

        mbgdml_pset = predictSet(pset_16mers[solv])
        E_mbgdml_pred, F_mbgdml_pred = mbgdml_pset.nbody_predictions(mbgdml_pset.models_order)
        E_mbgdml_error = E_mbgdml_pred[0] - E_ref

        F_mbgdml_error = F_mbgdml_pred - F_ref
        #F_mbgdml_rmse = np.sqrt(np.mean((F_mbgdml_error)**2))
        #F_rmse.append(F_mbgdml_rmse)
        F_mbgdml_mae = np.mean(np.abs(F_mbgdml_error))
        F_mae.append(F_mbgdml_mae)

        E_error.append(E_mbgdml_error)
        xaxis_labels.append(f'{solv_label}\nmbGDML')
        E_colors.append(solvent_colors[solv])
        print(f'{solv_label} mbGDML Energy Error: {E_mbgdml_error:.1f} kcal/mol')
        #print(f'{solv_label} mbGDML force RMSE with MP2: {F_mbgdml_rmse:.3f} kcal/mol/A\n')
        print(f'{solv_label} mbGDML force MAE with MP2: {F_mbgdml_mae:.3f} kcal/mol/A\n')


# Add data to plot.
marker_size = 5
line_width = 2.5

## Creating manual lollipop chart.
for i in range(len(E_error)):
    i_offset = 0.1

    # Energy
    e_x_value = i - i_offset
    e_y_value = abs(E_error[i])

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
ax1.set_xticks([i for i in range(len(E_error))])
ax1.set_xticklabels(xaxis_labels)

# y-axis for energy
ax1.set_ylabel('Absolute Energy Error (kcal/mol)')
y_tick_start, y_tick_end = ax1.get_ylim()
ax1.set_yticks(np.arange(0, y_tick_end, 1))
ax1.set_yticks(np.arange(0, y_tick_end, 0.5), minor=True)
ax1.set_ylim(ymin=0)

# y-axis for force rmse
#ax2.set_ylabel('Force RMSE (kcal/(mol $\AA$))')
ax2.set_ylabel('Force MAE (kcal/(mol $\AA$))')
y_tick_start, y_tick_end = ax2.get_ylim()
#ax2.set_yticks(np.arange(0, y_tick_end, 1))
#ax2.set_yticks(np.arange(0, y_tick_end, 0.5), minor=True)
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
ax1.legend(handles=legend_elements, loc='upper center', frameon=False)
    

plt_path = f'{save_dir}{plot_name}.png'

print(f'Saving {plt_path}')
plt.savefig(plt_path, dpi=1000)

