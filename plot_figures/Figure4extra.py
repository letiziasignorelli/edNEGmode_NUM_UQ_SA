import matplotlib.pyplot as plt
from prettyplot import *
import h5py
import os
import numpy as np
from numpy.linalg import norm

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


############   FIGURE3extra  ############
#   Plot convergence analysis for e_2   #
#########################################

types = ['sol', 'sol']
vars = ['K_se', 'V_se']
sim_folder_ref = 'data/simulation_outputs/NUM_reference'
sim_folder_cur = 'data/simulation_outputs/NUM_convergence_PHY'
solver_ref = 'LSODA1e-3PHY'
panel = np.array(['A', 'B', 'C', 'D'])

set_style("seaborn-paper")
fig = plt.figure()
axesA = plt.subplot(221)
axesB = plt.subplot(222, sharex=axesA, sharey=axesA)
axesC = plt.subplot(223, sharex=axesA)
axesD = plt.subplot(224, sharex=axesA, sharey=axesC)
axes = np.asarray([[axesA, axesB], [axesC, axesD]])

fig.set_figwidth(9)
fig.set_figheight(10)
normvalue = 2
t_step_cur = [0.0125, 0.025, 0.05, 0.1, 1, 1e1]
solver_cur = ['RK23', 'RK45', 'DOP853', 'BDF', 'Radau', 'LSODA']
markers = ['o', 'v', '*', 'x', 'D', 's']

print("--------------------------------------------------")
print('Errors norm: ' + str(normvalue) + ', Physiological')
print("--------------------------------------------------")

# Subplots for different variables
nx = 0
for var in vars:  
    ny = vars.index(var)

    print('Variable: '+ var)
    print("--------------------------------------------------")

    #Import reference
    path = os.path.join(sim_folder_ref, solver_ref + '.h5')
    hf = h5py.File(path, 'r') 
    t_ref = hf['time'][()]*1e-3
    sol_index = switch(types[ny],var)

    variable_ref = hf[types[ny]][(sol_index)]

    if var in concentrations:
        sol_index1 = switch('sol','V_se')  
        variable_ref = variable_ref / hf['sol'][(sol_index1)] *1e-3

    for solver in solver_cur:
        #Import current simulation
        if solver_cur.index(solver) < 3:
            path = os.path.join(sim_folder_cur, solver + '.h5')
        else:
            path = os.path.join(sim_folder_cur, solver + 'j.h5')
        hf = h5py.File(path, 'r') 

        temp = []
        norms = []

        for t_step in t_step_cur:
            
            if str(t_step) in hf.keys():   

                if t_step == np.inf:
                    temp.append(1e3)
                else:
                    temp.append(t_step)

                tt = hf[str(t_step)]['time'][()]*1e-3
                variable = hf[str(t_step)][types[ny]][(sol_index)]

                if var in concentrations:
                    variable = variable / hf[str(t_step)]['sol'][(sol_index1)] *1e-3

                comp_norm = norm(np.interp(t_ref,tt,variable)-variable_ref, normvalue)
                comp_norm /= norm(variable_ref, normvalue)

                norms.append(comp_norm)
        
        prettyPlot(temp, norms,
                color=solver_cur.index(solver),
                nr_colors=len(solver_cur),
                ax=axes[ny][nx],
                marker=markers[solver_cur.index(solver)],
                markersize=7,
                palette="tab10")
        
        print('Solver: '+ solver)
        print(["{:.3e}".format(norms[k]) for k in range(len(norms))])
        print("..................................................")
        
    axes[ny][nx].set_xscale('log')
    axes[ny][nx].set_yscale('log')
    # ABC
    axes[ny][nx].text(-0.05, 1.1, panel[nx+ny*(ny+1)], transform=axes[ny][nx].transAxes, fontsize=17, fontweight='bold', va='top', ha='right')

sim_folder_cur = 'data/simulation_outputs/NUM_convergence_PAT'
solver_ref = 'LSODA1e-3PAT'

print("--------------------------------------------------")
print('Errors norm: ' + str(normvalue) + ', Patological  ')
print("--------------------------------------------------")

# Subplots for different variables
nx = 1
for var in vars:  
    ny = vars.index(var)

    print('Variable: '+ var)
    print("--------------------------------------------------")

    #Import reference
    path = os.path.join(sim_folder_ref, solver_ref + '.h5')
    hf = h5py.File(path, 'r') 
    t_ref = hf['time'][()]*1e-3
    sol_index = switch(types[ny],var)

    variable_ref = hf[types[ny]][(sol_index)]

    if var in concentrations:
        sol_index1 = switch('sol','V_se')  
        variable_ref = variable_ref / hf['sol'][(sol_index1)] *1e-3

    for solver in solver_cur:
        #Import current simulation
        if solver_cur.index(solver) < 3:
            path = os.path.join(sim_folder_cur, solver + '.h5')
        else:
            path = os.path.join(sim_folder_cur, solver + 'j.h5')
        hf = h5py.File(path, 'r') 

        temp = []
        norms = []

        for t_step in t_step_cur:
            
            if str(t_step) in hf.keys():   

                if t_step == np.inf:
                    temp.append(1e3)
                else:
                    temp.append(t_step)

                tt = hf[str(t_step)]['time'][()]*1e-3
                variable = hf[str(t_step)][types[ny]][(sol_index)]

                if var in concentrations:
                    variable = variable / hf[str(t_step)]['sol'][(sol_index1)] *1e-3

                comp_norm = norm(np.interp(t_ref,tt,variable)-variable_ref, normvalue)
                comp_norm /= norm(variable_ref, normvalue)

                norms.append(comp_norm)
        
        prettyPlot(temp, norms,
                color=solver_cur.index(solver),
                nr_colors=len(solver_cur),
                ax=axes[ny][nx],
                marker=markers[solver_cur.index(solver)],
                markersize=7,
                palette="tab10")
        
        print('Solver: '+ solver)
        print(["{:.3e}".format(norms[k]) for k in range(len(norms))])
        print("..................................................")
        
    axes[ny][nx].set_xscale('log')
    axes[ny][nx].set_yscale('log')
    # ABC
    axes[ny][nx].text(-0.05, 1.1, panel[nx+ny*(ny+1)], transform=axes[ny][nx].transAxes, fontsize=17, fontweight='bold', va='top', ha='right')

# Set legend
set_legend(labels=solver_cur, bbox_to_anchor=(1,1.27), fontsize=12, ncol=6, frameon=False, columnspacing=1.5)

# Axes labels
axesC.set_xlabel(r'$\Delta t_\mathrm{max}$ $\mathrm{[ms]}$', size = 13)
axesD.set_xlabel(r'$\Delta t_\mathrm{max}$ $\mathrm{[ms]}$', size = 13)

axesA.set_ylabel(r'$e_2^{\mathrm{[K^+]_{se}}}$', size=13)
axesC.set_ylabel(r'$e_2^{\mathrm{V_{se}}}$', size=13)

# Set titles
axesA.set_title('Physiological', size=20, pad=17)
axesB.set_title('Pathological', size=20, pad=17)

plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.25, hspace=0.4)
plt.savefig('plot_figures/figures/figure4extra.eps', format='eps')
plt.show()
