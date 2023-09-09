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


### FIGURE3 ###
types = ['phi', 'sol', 'sol']
vars = ['phi_msn', 'K_se', 'V_se']
sim_folder = 'data/simulation_outputs/NUM_reference'
solvers = ['LSODA1e-3REST', 'LSODA1e-3PHY','LSODA1e-3PAT']
t_step = 1e-3
panel = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])

set_style("seaborn-paper")
fig = plt.figure()
axesA = plt.subplot(331)
axesB = plt.subplot(332)
axesC = plt.subplot(333, sharey=axesB)
axesD = plt.subplot(334)
axesE = plt.subplot(335)
axesF = plt.subplot(336)
axesG = plt.subplot(337)
axesH = plt.subplot(338)
axesI = plt.subplot(339, sharey=axesH)
axes = np.asarray([[axesA, axesB, axesC], [axesD, axesE, axesF], [axesG, axesH, axesI]])

fig.set_figwidth(10)
fig.set_figheight(7.5)

for i in range(0, 3*len(vars)):
    nx = i % 3
    ny = int(np.floor(i/float(3)))

    if i < 3*len(vars):
        path = os.path.join(sim_folder, solvers[nx] + '.h5')
        hf = h5py.File(path, 'r') 
        t_ref = hf['time'][()]*1e-3

        sol_index = switch(types[ny],vars[ny])  
        variable_ref = hf[types[ny]][(sol_index)]  

        if vars[ny] in concentrations:
            sol_index1 = switch('sol','V_se')  
            variable_ref = variable_ref / hf['sol'][(sol_index1)] *1e-3

        if vars[ny] in volumes:
            variable_ref = (variable_ref - variable_ref[0])/variable_ref[0]*100

        prettyPlot(t_ref, variable_ref,
                    color=ny,
                    nr_colors=3*len(vars),
                    ax=axes[ny][nx],
                    palette="tab10")

        axes[ny][nx].set_xlim([min(t_ref), max(t_ref)])

        # ABC
        axes[ny][nx].text(-0.05, 1.2, panel[i], transform=axes[ny][nx].transAxes, fontsize=17, fontweight='bold', va='top', ha='right')
        axes[ny][nx].ticklabel_format(useMathText=True)

# Axes limits
axes[0,0].set_ylim([-67.5,-63])
axes[1,0].set_ylim([3.53,3.65])
axes[2,0].set_ylim([-1.45,0.05])

# Axes labels
axes[0,0].set_ylabel(r'$\phi_\mathrm{msn}$ $\mathrm{[mV]}$', size = 13)
axes[1,0].set_ylabel(r'$\mathrm{[K^+]_{se}}$ $\mathrm{[mM]}$', size = 13)
axes[2,0].set_ylabel(r'$\Delta \mathrm{V_{se}}$ $[\%]$', size = 13)

axes[2,0].set_xlabel(r'$\mathrm{time}$ $\mathrm{[s]}$', size = 13)
axes[2,1].set_xlabel(r'$\mathrm{time}$ $\mathrm{[s]}$', size = 13)
axes[2,2].set_xlabel(r'$\mathrm{time}$ $\mathrm{[s]}$', size = 13)

# Set titles
axes[0,0].set_title('Resting state', size=20, pad=12)
axes[0,1].set_title('Physiological', size=20, pad=12)
axes[0,2].set_title('Pathological', size=20, pad=12)

plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.25, hspace=0.45)
plt.savefig('plot_figures/figures/figure3.eps', format='eps')
plt.show()



