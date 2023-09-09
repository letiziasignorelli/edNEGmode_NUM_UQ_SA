import matplotlib.pyplot as plt
from prettyplot import *
import h5py
import os
import numpy as np
# TYPE CHOICE
'''
type choice =   'sol'   : solution (Number of ions, Gating variables, Volumes)
                'phi'   : membrane potentials
                'E'     : reversal potentials
'''

# VAR CHOICE
'''
for type = sol
    --> choose: 'Na_sn', 'Na_se', 'Na_sg', 'Na_dn', 'Na_de', 'Na_dg', 'K_sn', 'K_se', 'K_sg', 'K_dn', 'K_de', 'K_dg', 'Cl_sn', 'Cl_se', 
                'Cl_sg', 'Cl_dn', 'Cl_de', 'Cl_dg', 'Ca_sn', 'Ca_se', 'Ca_dn', 'Ca_de', 'n', 'h', 's', 'c', 'q', 'z', 'V_sn', 'V_se', 'V_sg', 'V_dn', 'V_de', 'V_dg'
for type = phi
    --> choose: 'phi_sn', 'phi_se', 'phi_sg', 'phi_dn', 'phi_de', 'phi_dg', 'phi_msn', 'phi_mdn', 'phi_msg', 'phi_mdg'
for type = E
    --> choose: 'E_Na_sn', 'E_Na_sg', 'E_Na_dn', 'E_Na_dg', 'E_K_sn', 'E_K_sg', 'E_K_dn', 'E_K_dg', 'E_Cl_sn', 'E_Cl_sg', 'E_Cl_dn', 'E_Cl_dg', 'E_Ca_sn', 'E_Ca_dn'
'''

def compute_local_maxima(t,variable,interval):
    index_left = np.where(t >= interval[0])[0][0]
    index_right = np.where(t >= interval[1])[0][0]

    local_time = t[index_left:index_right]
    local_variable = variable[index_left:index_right]

    firing = False
    i_max = []
    control_firing = False

    for i in range(len(local_time)):
        
        if firing == False and local_variable[i] > -10:
            firing = True
        if firing==True and control_firing == False and local_variable[i-1]<local_variable[i] and local_variable[i]>local_variable[i+1]:
            i_max.append(i)
            control_firing = True
        elif firing == True and local_variable[i] < -10:
            firing = False
            control_firing = False

    t_val = [local_time[k] for k in i_max]
    var_val = [local_variable[k] for k in i_max]

    return t_val, var_val

### FIGURE5 ###
sim_folder = 'data/simulation_outputs/NUM_convergence_PHY'
t_steps = [0.0125, 0.025, 0.05, 0.1, 1, 1e1]
solvers = ['RK23', 'RK45', 'DOP853', 'BDF', 'Radau', 'LSODA']
markers = ['o', 'v', '*', 'x', 'D', 's']

set_style("seaborn-paper")
fig = plt.figure()
axes = plt.subplot(111)
sol_index = switch('phi','phi_msn')

fig.set_figwidth(7.5)
fig.set_figheight(5)

for solver in solvers:
    #Import current simulation
    if solvers.index(solver) < 3:
        path = os.path.join(sim_folder, solver + '.h5')
    else:
        path = os.path.join(sim_folder, solver + 'j.h5')
    hf = h5py.File(path, 'r') 

    temp = []
    ap_number = []
    last_ap = []

    for t_step in t_steps:
        
        if str(t_step) in hf.keys():   

            if t_step == np.inf:
                temp.append(1e3)
            else:
                temp.append(t_step)

            tt = hf[str(t_step)]['time'][()] *1e-3
            variable = hf[str(t_step)]['phi'][(sol_index)]
            t_val, var_val = compute_local_maxima(tt,variable,[1,5])

            # Compute N_AP
            ap_number.append(len(var_val))

            # Compute T_lAP
            last_ap.append(t_val[-1])
    
    prettyPlot(temp, last_ap,
            color=solvers.index(solver),
            nr_colors=len(solvers),
            ax=axes,
            marker=markers[solvers.index(solver)],
            markersize=7,
            palette="tab10")
    
    print('Solver: '+ solver)
    print('N_AP = ' + str(ap_number))
    print('T_lAP = ' + str(last_ap))
    print("..................................................")

axes.set_xscale('log')
set_legend(labels=solvers, bbox_to_anchor=(1,0.75), fontsize=12, frameon=False, labelspacing=1)

# Axes labels
axes.set_xlabel(r'$\Delta t_\mathrm{max}$ $\mathrm{[ms]}$', size = 13)
axes.set_ylabel(r'$T_\mathrm{lAP}$ $\mathrm{[s]}$', size=13)

plt.tight_layout()
plt.savefig('plot_figures/figures/figure5.eps', format='eps')
plt.show()


