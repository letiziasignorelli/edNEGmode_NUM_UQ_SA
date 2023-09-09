import numpy as np
from edNEGmodel_rescaled.edNEGmodel_rescaled import *
from functions.initial_values_rescaled import *
from functions.solve_edNEGmodel_rescaled import *
import os

print('*************************************************')
print('*        Computing referece simulation          *')
print('*************************************************')

# Rescaled test parameters in resting conditions
I_stim = 0.             # [uA]
alpha = 2
t_dur = 1e4             # [ms]
t_span = (0, t_dur)
stim_start = 0.
stim_end = 0.

# Save path
# checking if the directory exist and create it if it doesn't
savepath = "data/simulation_outputs/NUM_reference"
file_name = "LSODA1e-3REST.h5"
if not os.path.exists(savepath):
    os.makedirs(savepath)
path = os.path.join(savepath, file_name)

# Choose simulation setting
solver = 'LSODA'
t_step = 1e-3

# Solve
solve_edNEGmodel_reference(I_stim, alpha, t_span, stim_start, stim_end, path, solver, t_step, jacobian=True)


# Rescaled test parameters in physiological conditions
I_stim = 7.2e-5         # [uA]
alpha = 2
t_dur = 1e4             # [ms]
t_span = (0, t_dur)
stim_start = 1e3
stim_end = 5e3         

# Save path
file_name = "LSODA1e-3PHY.h5"
path = os.path.join(savepath, file_name)
# Solve
solve_edNEGmodel_reference(I_stim, alpha, t_span, stim_start, stim_end, path, solver, t_step, jacobian=True)


# Rescaled test parameters in pathological conditions
I_stim = 15e-5          # [uA]
alpha = 2
t_dur = 1e4             # [ms]
t_span = (0, t_dur)
stim_start = 1e3
stim_end = 8e3  

# Save path
file_name = "LSODA1e-3PAT.h5"
path = os.path.join(savepath, file_name)
# Solve
solve_edNEGmodel_reference(I_stim, alpha, t_span, stim_start, stim_end, path, solver, t_step, jacobian=True)