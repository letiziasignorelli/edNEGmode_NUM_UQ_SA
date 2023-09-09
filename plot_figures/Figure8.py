import uncertainpy as un
import numpy as np
import matplotlib.pyplot as plt
from prettyplot import * 
import os
import h5py

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

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel(r'$\hat{\sigma}$ $[\%]$')

### FIGURE8 ###
# Import nominal data
sim_folder = 'data/simulation_outputs/UQSA_dyn_state'
path = os.path.join(sim_folder, 'nominal.h5')
hf = h5py.File(path, 'r') 
t_ref = hf['time'][()]*1e-3

sol_index = switch('phi','phi_msn')  
variable_ref = hf['phi'][(sol_index)] 
t_val, var_val = compute_local_maxima(t_ref,variable_ref,[0.2,2.8])
nr_spikes_nominal = np.full(254, len(var_val))
last_spike_nominal = np.full(254, t_val[-1])
time_before_first_spike_nominal = np.full(254, (t_val[0] - 0.2)*1e3)

# Import UQ data
data1 = un.Data()
data1.load(filename='data/simulation_outputs/UQSA_dyn_state/phi_msn_1/nr_spikes.h5')
data5 = un.Data()
data5.load(filename='data/simulation_outputs/UQSA_dyn_state/phi_msn_5/nr_spikes.h5')
data10 = un.Data()
data10.load(filename='data/simulation_outputs/UQSA_dyn_state/phi_msn_10/nr_spikes.h5')

nr_spikes1 = data1['nr_spikes']['evaluations']
nr_spikes5 = data5['nr_spikes']['evaluations']
nr_spikes10 = data10['nr_spikes']['evaluations']

last_spike1 = data1['last_spike']['evaluations']*1e-3
last_spike5 = data5['last_spike']['evaluations']*1e-3
last_spike10 = data10['last_spike']['evaluations']*1e-3

time_before_first_spike1 = data1['time_before_first_spike']['evaluations']
time_before_first_spike5 = data5['time_before_first_spike']['evaluations']
time_before_first_spike10 = data10['time_before_first_spike']['evaluations']

nr_spikes = [nr_spikes_nominal, nr_spikes1, nr_spikes5, nr_spikes10]
last_spike = [last_spike_nominal, last_spike1, last_spike5, last_spike10]
time_before_first_spike = [ time_before_first_spike_nominal, time_before_first_spike1, time_before_first_spike5, time_before_first_spike10]

# Normalized data
nominal = np.full(254,0)

nr_spikes1_n = (data1['nr_spikes']['evaluations'] - data1['nr_spikes']['mean'])/np.sqrt(data1['nr_spikes']['variance'])
nr_spikes5_n = (data5['nr_spikes']['evaluations'] - data5['nr_spikes']['mean'])/np.sqrt(data5['nr_spikes']['variance'])
nr_spikes10_n = (data10['nr_spikes']['evaluations'] - data10['nr_spikes']['mean'])/np.sqrt(data10['nr_spikes']['variance'])

last_spike1_n = (data1['last_spike']['evaluations'] - data1['last_spike']['mean'])/np.sqrt(data1['last_spike']['variance'])
last_spike5_n = (data5['last_spike']['evaluations'] - data5['last_spike']['mean'])/np.sqrt(data5['last_spike']['variance'])
last_spike10_n = (data10['last_spike']['evaluations'] - data10['last_spike']['mean'])/np.sqrt(data10['last_spike']['variance'])

time_before_first_spike1_n = (data1['time_before_first_spike']['evaluations'] - data1['time_before_first_spike']['mean'])/np.sqrt(data1['time_before_first_spike']['variance'])
time_before_first_spike5_n = (data5['time_before_first_spike']['evaluations'] - data5['time_before_first_spike']['mean'])/np.sqrt(data5['time_before_first_spike']['variance'])
time_before_first_spike10_n = (data10['time_before_first_spike']['evaluations'] - data10['time_before_first_spike']['mean'])/np.sqrt(data10['time_before_first_spike']['variance'])

nr_spikes_n = [nominal, nr_spikes1_n, nr_spikes5_n, nr_spikes10_n]
last_spike_n = [nominal, last_spike1_n, last_spike5_n, last_spike10_n]
time_before_first_spike_n = [ nominal, time_before_first_spike1_n, time_before_first_spike5_n, time_before_first_spike10_n]

# Setup figure
set_style("seaborn-paper")
fig = plt.figure()
ax = fig.add_gridspec(3, 3)
axesA = fig.add_subplot(ax[0,0])
axesB = fig.add_subplot(ax[0,1])
axesC = fig.add_subplot(ax[0,2])
axesD = fig.add_subplot(ax[1,0])
axesE = fig.add_subplot(ax[1,1])
axesF = fig.add_subplot(ax[1,2])
axesG = fig.add_subplot(ax[2, 0:3])

fig.set_figwidth(9)
fig.set_figheight(8)

colors = ["#e377c2", "#bcbd22", "#17becf"]

# Distributions
axesA.hist(nr_spikes5, bins=30, color=colors[0])
axesA.axvline(nr_spikes_nominal[0], color='r', linestyle='dashed', linewidth=1)
axesB.hist(last_spike5, bins=30, color=colors[1])
axesB.axvline(last_spike_nominal[0], color='r', linestyle='dashed', linewidth=1)
axesC.hist(time_before_first_spike5, bins=30, color=colors[2])
axesC.axvline(time_before_first_spike_nominal[0], color='r', linestyle='dashed', linewidth=1)

axesA.set_xlabel(r"$N_\mathrm{AP}$", size = 13)
axesB.set_xlabel(r"$T_\mathrm{lAP}$ $\mathrm{[s]}$", size = 13)
axesC.set_xlabel(r'$T_\mathrm{bFAP}$ $\mathrm{[ms]}$', size = 13)

axesA.set_yticks([])
axesB.set_yticks([])
axesC.set_yticks([])

axesA.text(39,18.5, 'Baseline', fontsize=9)
axesB.text(2.701, 14, 'Baseline', fontsize=9)
axesC.text(8.01,20.2, 'Baseline', fontsize=9)

# Violin plots
# N_AP
parts_nap = axesD.violinplot(
        nr_spikes, showmeans=False, showmedians=False,
        showextrema=False)
for pc in parts_nap['bodies']:
    pc.set_facecolor('#e377c2')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(nr_spikes, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(nr_spikes, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
axesD.scatter(inds[1:], medians[1:], marker='o', color='white', s=10, zorder=3)
axesD.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3)
axesD.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# T_lAP
parts_fdss = axesE.violinplot(
        last_spike, showmeans=False, showmedians=False,
        showextrema=False)
for pc in parts_fdss['bodies']:
    pc.set_facecolor('#bcbd22')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(last_spike, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(last_spike, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
axesE.scatter(inds[1:], medians[1:], marker='o', color='white', s=10, zorder=3)
axesE.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3)
axesE.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# T_bFS
parts_tbfs = axesF.violinplot(
        time_before_first_spike, showmeans=False, showmedians=False,
        showextrema=False)
for pc in parts_tbfs['bodies']:
    pc.set_facecolor('#17becf')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(time_before_first_spike, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(time_before_first_spike, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
axesF.scatter(inds[1:], medians[1:], marker='o', color='white', s=10, zorder=3)
axesF.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3)
axesF.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# Normalized all
# N_AP
quartile1, medians, quartile3 = np.percentile(nr_spikes_n, [25, 50, 75], axis=1)
inds = np.arange(1, len(medians) + 1)

parts_nap_n = axesG.violinplot(
        nr_spikes_n,  positions=inds-0.22, widths=0.2, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts_nap_n['bodies']:
    pc.set_facecolor('#e377c2')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(nr_spikes_n, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

axesG.scatter(inds[1:]-0.22, medians[1:], marker='o', color='white', s=10, zorder=3)
axesG.vlines(inds-0.22, quartile1, quartile3, color='k', linestyle='-', lw=3)
axesG.vlines(inds-0.22, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# f_DSS
quartile1, medians, quartile3 = np.percentile(last_spike_n, [25, 50, 75], axis=1)
inds = np.arange(1, len(medians) + 1)

parts_fdss_n = axesG.violinplot(
        last_spike_n,  positions=inds, widths=0.2, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts_fdss_n['bodies']:
    pc.set_facecolor('#bcbd22')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(last_spike_n, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

axesG.scatter(inds[1:], medians[1:], marker='o', color='white', s=10, zorder=3)
axesG.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3)
axesG.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# T_bFS
quartile1, medians, quartile3 = np.percentile(time_before_first_spike_n, [25, 50, 75], axis=1)
inds = np.arange(1, len(medians) + 1)

parts_tbfs_n = axesG.violinplot(
        time_before_first_spike_n,  positions=inds+0.22, widths=0.2, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts_tbfs_n['bodies']:
    pc.set_facecolor('#17becf')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(time_before_first_spike_n, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

axesG.scatter(inds[1:]+0.22, medians[1:], marker='o', color='white', s=10, zorder=3)
axesG.vlines(inds+0.22, quartile1, quartile3, color='k', linestyle='-', lw=3)
axesG.vlines(inds+0.22, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

axesG.legend([parts_nap_n['bodies'][0], parts_fdss_n['bodies'][0], parts_tbfs_n['bodies'][0]],[r"$N_\mathrm{AP}^\mathrm{(norm)}$", r"$T_\mathrm{lAP}^\mathrm{(norm)}$", r"$T_\mathrm{bFAP}^\mathrm{(norm)}$"], 
              fontsize=12 ,bbox_to_anchor=(0.2,1.2), frameon=False, labelspacing=0.1)

# set style for the axes
labels = ['0', '1', '5', '10']
for ax in [axesD, axesE, axesF, axesG]:
    set_axis_style(ax, labels)


axesD.set_ylabel(r"$N_\mathrm{AP}$", size=13)
axesE.set_ylabel(r"$T_\mathrm{lAP}$ $\mathrm{[s]}$", size=13)
axesF.set_ylabel(r'$T_\mathrm{bFAP}$ $\mathrm{[ms]}$', size=13)

# ABC
axesA.text(-0.05, 1.2, 'A', transform=axesA.transAxes, fontsize=17, fontweight='bold', va='top', ha='right')
axesB.text(-0.05, 1.2, 'B', transform=axesB.transAxes, fontsize=17, fontweight='bold', va='top', ha='right')
axesC.text(-0.05, 1.2, 'C', transform=axesC.transAxes, fontsize=17, fontweight='bold', va='top', ha='right')
axesD.text(-0.05, 1.2, 'D', transform=axesD.transAxes, fontsize=17, fontweight='bold', va='top', ha='right')
axesE.text(-0.05, 1.2, 'E', transform=axesE.transAxes, fontsize=17, fontweight='bold', va='top', ha='right')
axesF.text(-0.05, 1.2, 'F', transform=axesF.transAxes, fontsize=17, fontweight='bold', va='top', ha='right')
axesG.text(0, 1.2, 'G', transform=axesG.transAxes, fontsize=17, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.55)
plt.savefig('plot_figures/figures/figure8.eps', format='eps')
plt.show()