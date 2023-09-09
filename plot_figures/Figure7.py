import matplotlib.pyplot as plt
from prettyplot import *
import h5py
import os

### FIGURE7 ###
types = ['phi', 'sol']
vars = ['phi_msn', 'K_se']
sim_folder = 'data/simulation_outputs/UQSA_dyn_state'

path = os.path.join(sim_folder, 'nominal.h5')
hf = h5py.File(path, 'r') 
t_ref = hf['time'][()]*1e-3

# Plot
set_style("seaborn-paper")
fig = plt.figure()
axesA = plt.subplot(121)
axesB = plt.subplot(122, sharex=axesA)

fig.set_figwidth(8)
fig.set_figheight(4)

# Panel A
sol_index = switch(types[0], vars[0])  
variable = hf[types[0]][(sol_index)]  

prettyPlot(t_ref, variable,
                    color=0,
                    nr_colors=2,
                    ax=axesA,
                    palette="tab10")

axesA.set_xlim([min(t_ref), max(t_ref)])
axesA.ticklabel_format(useMathText=True)
axesA.set_ylabel(r'$\phi_\mathrm{msn}$ $\mathrm{[mV]}$', size = 13)
axesA.set_xlabel(r'$\mathrm{time}$ $\mathrm{[s]}$', size = 13)

# Panel B
sol_index = switch(types[1], vars[1]) 
sol_index1 = switch('sol','V_se')  
variable = hf[types[1]][(sol_index)]  
variable = variable / hf['sol'][(sol_index1)] *1e-3

prettyPlot(t_ref, variable,
                    color=1,
                    nr_colors=2,
                    ax=axesB,
                    palette="tab10")

axesB.set_xlim([min(t_ref), max(t_ref)])
axesB.ticklabel_format(useMathText=True)
axesB.set_ylabel(r'$\mathrm{[K^+]_{se}}$ $\mathrm{[mM]}$', size = 13)
axesB.set_xlabel(r'$\mathrm{time}$ $\mathrm{[s]}$', size = 13)
plt.tight_layout()
plt.savefig('plot_figures/figures/figure7.eps', format='eps')
plt.show()