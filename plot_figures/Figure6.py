import matplotlib.pyplot as plt
from prettyplot import *
import h5py
import os
import numpy as np
from numpy.linalg import norm
from matplotlib.ticker import LogLocator, ScalarFormatter, StrMethodFormatter

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
def elapsed_times(solvers, t_steps, sim_folder, axx):
    '''
    Compute and plot norm of the difference between different simulations
    '''
    for solver in solvers:
        # print(solver)
        #Import current simulation
        if solvers.index(solver) < 3:
            path = os.path.join(sim_folder, solver + '.h5')
        else:
            path = os.path.join(sim_folder, solver + 'j.h5')
        hf = h5py.File(path, 'r') 
        print(solver)
        temp = []
        times = []

        for t_step in t_steps:
            if str(t_step) in hf.keys():

                if t_step == np.inf:
                    temp.append(1e3)
                else:
                    temp.append(t_step)

                times.append(hf[str(t_step)]['elapsed_time'][()])

        prettyPlot(temp, times,
                color=solvers.index(solver),
                nr_colors=len(solvers),
                ax=axx,
                marker=markers[solvers.index(solver)],
                markersize=7,
                palette="tab10")
        axx.set_xscale('log')
        axx.set_yscale('log')
        print(times)
        print("--------------------------------------------------")

    return times

#############     FIGURE5      #############
# Plot efficiency analysis (elapsed times) #
############################################
set_style("seaborn-paper")
fig = plt.figure()
axesA = plt.subplot(121)
axesB = plt.subplot(122, sharex=axesA, sharey=axesA)

fig.set_figwidth(10)
fig.set_figheight(5)

t_steps = [0.0125, 0.025, 0.05, 0.1, 1, 1e1]
solvers = ['RK23', 'RK45', 'DOP853', 'BDF', 'Radau', 'LSODA']
markers = ['o', 'v', '*', 'x', 'D', 's']

# Physiological conditions
print("--------------------------------------------------")
print('Elapsed times, Physiological')
print("--------------------------------------------------")

sim_folder = 'data/simulation_outputs/NUM_convergence_PHY'
times = elapsed_times(solvers, t_steps, sim_folder, axesA)

# Pathological conditions
print("--------------------------------------------------")
print('Elapsed times, Pathological')
print("--------------------------------------------------")
sim_folder = 'data/simulation_outputs/NUM_convergence_PAT'
times = elapsed_times(solvers, t_steps, sim_folder, axesB)

# Set legend
set_legend(labels=solvers, bbox_to_anchor=(-0.1,1), fontsize=12, frameon=False, labelspacing=1)

# Axes labels
axesA.set_xlabel(r'$\Delta t_\mathrm{max}$ $\mathrm{[ms]}$', size = 13)
axesB.set_xlabel(r'$\Delta t_\mathrm{max}$ $\mathrm{[ms]}$', size = 13)

axesA.set_ylabel(r'$\mathrm{Elapsed \ time \ [s]}$', size=13)

# Set titles
axesA.set_title('Physiological', size=20, pad=17)
axesB.set_title('Pathological', size=20, pad=17)

plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.25, hspace=0.4)
plt.savefig('plot_figures/figures/figure6.eps', format='eps')
plt.show()
