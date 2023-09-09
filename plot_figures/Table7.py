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

def elapsed_times(solvers, t_steps, sim_folder):
    '''
    Compute and plot norm of the difference between different simulations
    '''
    # Setup 1st timestep for plot if max_step=np.inf
    temp = np.copy(t_steps)
    if temp[0] == np.inf:    
        temp[0] = 1e3
    times = np.zeros([len(solvers), len(t_steps)])
    for solver in solvers:
        print(solver)
        #Import current simulation
        path = os.path.join(sim_folder, solver + '.h5')
        hf = h5py.File(path, 'r') 

        for t_step in t_steps:
            if str(t_step) in hf.keys():
                times[solvers.index(solver), t_steps.index(t_step)] = hf[str(t_step)]['elapsed_time'][()]
                
        print(times[solvers.index(solver), :])
        print("--------------------------------------------------")

    return times

### TABLE5 ###
# Print elapsed times of implicit solver - Analytical Jacobian (A) vd Finite-Differences approximation (FD)
sim_folder = 'data/simulation_outputs/NUM_convergence_PHY'
t_steps = [0.0125, 0.025, 0.05, 0.1, 1, 1e1]
solvers = ['BDFj','BDF', 'Radauj', 'Radau', 'LSODAj','LSODA']

print("**************************************************")
print('Elapsed times in Physiological conditions  ')
print("**************************************************")
times = elapsed_times(solvers, t_steps, sim_folder)

print("**************************************************")
print('Gained efficiency using analytical Jacobian  [%]  ')
print("**************************************************")
for k in range(3):
    print(solvers[2*k+1])
    efficiency = []
    for i in range(len(t_steps)):
        if times[2*k,i] != 0.:
            efficiency.append(np.abs((times[2*k,i] - times[2*k+1,i]) / times[2*k+1,i] * 100))
    
    print(efficiency)