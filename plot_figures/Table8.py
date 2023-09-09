import matplotlib.pyplot as plt
from prettyplot import *
import h5py
import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


### TABLE8 ###
savepath = 'data/simulation_outputs/UQSA_rest_state'
file_name = "UQSA_rest_state.h5"
path = os.path.join(savepath, file_name)
hf = h5py.File(path, 'r')

# Open results
output_samples = hf['output_samples'][()]
first_order_indices = hf['S1'][()]
total_order_indices = hf['ST'][()]

output_list = [r'$\phi_\mathrm{msn}$', r'$\phi_\mathrm{mdn}$', r'$\phi_\mathrm{msg}$', r'$\phi_\mathrm{mdg}$', r'$\mathrm{[K^+]_{se}}$', r'$\mathrm{[K^+]_{de}}$']

print(output_list)
print(total_order_indices)

# # Plot
# set_style("seaborn-paper")
# fig = plt.figure()
# # axesA = plt.subplot(121)
# axesB = plt.subplot(111)

# fig.set_figwidth(5.5)
# fig.set_figheight(5)

# # First-order Sobol indices heatmap
# axesA.imshow(first_order_indices, sns.color_palette("coolwarm", as_cmap=True), aspect='auto')
# axesA.ticklabel_format(useMathText=True)
# axesA.set_xticks(np.arange(2))     #, rotation=45, ha='right')
# axesA.set_xticklabels(['leak', 'dynamic'], color='black')
# axesA.set_yticks(np.arange(len(output_list)))
# axesA.set_yticklabels(output_list, color='black')
# axesA.set_xlabel('Input Variables')
# axesA.set_ylabel('Output Variables')
# axesA.set_title('First-Order Sobol Indices')

# # Total-order Sobol indices heatmap
# im = axesB.imshow(total_order_indices, sns.color_palette("coolwarm", as_cmap=True), aspect='auto')
# axesB.set_xticks(np.arange(2))     #, rotation=45, ha='right')
# axesB.set_xticklabels(['leak', 'dynamic'], color='black', size = 13)
# axesB.set_yticks(np.arange(len(output_list)))
# axesB.set_yticklabels(output_list, color='black', size = 13)
# axesB.set_xlabel('Input Variables')
# axesB.set_ylabel('Output Variables')
# axesB.set_title('Total-Order Sobol Indices', size = 13)

# # Create a colorbar for the total-order Sobol indices heatmap
# divider = make_axes_locatable(axesB)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)

# # Adjust the layout and spacing between subplots
# plt.tight_layout()
# plt.savefig('plot_figures/figures/figure6.eps', format='eps')
# plt.show()