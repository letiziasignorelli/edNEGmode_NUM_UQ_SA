from edNEGmodel_rescaled.edNEGmodel_parameters import *
from scipy.integrate import solve_ivp
from functions.initial_values_rescaled import *

################################################################################
# List of all possible configurations supported by the code UQSA_rest_state.py #
################################################################################

# Define the problem setup

# Choose sigma_hat
ic = 0.15

# Setup problem parameters distributions and groups
problem_complete = {
    'num_vars': 16,
    'names': [  'g_Na_leak_n',
                'g_K_leak_n',
                'g_Cl_leak_n',
                'g_Na' ,      
                'g_DR' ,         
                'g_Ca' ,       
                'g_AHP',         
                'g_C'   ,      
                'g_Na_leak_g' ,  
                'g_K_IR' ,    
                'g_Cl_leak_g' , 
                'rho_n',        
                'U_kcc2' ,  
                'U_nkcc1' ,
                'U_Cadec' ,      
                'rho_g'],
    'bounds': [       
                [0.0246 * (1 - ic), 0.0246 * (1 + ic)],
                [0.0245 * (1 - ic), 0.0245 * (1 + ic)],
                [0.1 * (1 - ic), 0.1 * (1 + ic)],
                [30. * (1 - ic), 30. * (1 + ic)],
                [15. * (1 - ic), 15. * (1 + ic)],
                [11.8 * (1 - ic), 11.8 * (1 + ic)], 
                [.8 * (1 - ic), .8 * (1 + ic)],
                [15. * (1 - ic), 15. * (1 + ic)], 
                [.1 * (1 - ic), .1 * (1 + ic)],
                [1.696 * (1 - ic), 1.696 * (1 + ic)],
                [0.05 * (1 - ic), 0.05 * (1 + ic)], 
                [1.87e-4 * (1 - ic), 1.87e-4 * (1 + ic)],
                [1.49e-5 * (1 - ic), 1.49e-5 * (1 + ic)], 
                [2.33e-5 * (1 - ic), 2.33e-5 * (1 + ic)], 
                [0.075 * (1 - ic), 0.075 * (1 + ic)],
                [1.12e-4 * (1 - ic),1.12e-4 * (1 + ic)],
              ],
    'dists':['unif','unif','unif','unif','unif','unif','unif','unif',\
              'unif','unif','unif','unif','unif','unif','unif','unif'],
    'groups': None
    }

problem_leak = {
    'num_vars': 11,
    'names': [  'g_Na_leak_n',
                'g_K_leak_n',
                'g_Cl_leak_n',    
                'g_Na_leak_g' ,  
                'g_K_IR' ,    
                'g_Cl_leak_g' , 
                'rho_n',        
                'U_kcc2' ,  
                'U_nkcc1' ,
                'U_Cadec' ,      
                'rho_g'],
    'bounds': [       
                [0.0246 * (1 - ic), 0.0246 * (1 + ic)],
                [0.0245 * (1 - ic), 0.0245 * (1 + ic)],
                [0.1 * (1 - ic), 0.1 * (1 + ic)],
                [.1 * (1 - ic), .1 * (1 + ic)],
                [1.696 * (1 - ic), 1.696 * (1 + ic)],
                [0.05 * (1 - ic), 0.05 * (1 + ic)], 
                [1.87e-4 * (1 - ic), 1.87e-4 * (1 + ic)],
                [1.49e-5 * (1 - ic), 1.49e-5 * (1 + ic)], 
                [2.33e-5 * (1 - ic), 2.33e-5 * (1 + ic)], 
                [0.075 * (1 - ic), 0.075 * (1 + ic)],
                [1.12e-4 * (1 - ic),1.12e-4 * (1 + ic)],
              ],
    'dists':['unif','unif','unif','unif','unif','unif','unif','unif','unif','unif','unif'],
    'groups': None
    }

problem_dynamics = {
    'num_vars': 5,
    'names': [  'g_Na' ,      
                'g_DR' ,         
                'g_Ca' ,       
                'g_AHP',         
                'g_C'],
    'bounds': [ 
                [30. * (1 - ic), 30. * (1 + ic)],
                [15. * (1 - ic), 15. * (1 + ic)],
                [11.8 * (1 - ic), 11.8 * (1 + ic)], 
                [.8 * (1 - ic), .8 * (1 + ic)],
                [15. * (1 - ic), 15. * (1 + ic)]
              ],
    'dists':['unif','unif','unif','unif','unif'],
    'groups': None
    }

problem_complete_grouped = {
    'num_vars': 16,
    'names': [  'g_Na_leak_n',
                'g_K_leak_n',
                'g_Cl_leak_n',
                'g_Na' ,      
                'g_DR' ,         
                'g_Ca' ,       
                'g_AHP',         
                'g_C'   ,      
                'g_Na_leak_g' ,  
                'g_K_IR' ,    
                'g_Cl_leak_g' , 
                'rho_n',        
                'U_kcc2' ,  
                'U_nkcc1' ,
                'U_Cadec' ,      
                'rho_g'],
    'bounds': [       
                [0.0246 * (1 - ic), 0.0246 * (1 + ic)],
                [0.0245 * (1 - ic), 0.0245 * (1 + ic)],
                [0.1 * (1 - ic), 0.1 * (1 + ic)],
                [30. * (1 - ic), 30. * (1 + ic)],
                [15. * (1 - ic), 15. * (1 + ic)],
                [11.8 * (1 - ic), 11.8 * (1 + ic)], 
                [.8 * (1 - ic), .8 * (1 + ic)],
                [15. * (1 - ic), 15. * (1 + ic)], 
                [.1 * (1 - ic), .1 * (1 + ic)],
                [1.696 * (1 - ic), 1.696 * (1 + ic)],
                [0.05 * (1 - ic), 0.05 * (1 + ic)], 
                [1.87e-4 * (1 - ic), 1.87e-4 * (1 + ic)],
                [1.49e-5 * (1 - ic), 1.49e-5 * (1 + ic)], 
                [2.33e-5 * (1 - ic), 2.33e-5 * (1 + ic)], 
                [0.075 * (1 - ic), 0.075 * (1 + ic)],
                [1.12e-4 * (1 - ic),1.12e-4 * (1 + ic)],
              ],
    'dists':['unif','unif','unif','unif','unif','unif','unif','unif',\
              'unif','unif','unif','unif','unif','unif','unif','unif'],
    'groups': ['leak','leak','leak','dyn','dyn','dyn','dyn','dyn',\
              'leak','leak','leak','leak','leak','leak','leak','leak']
    }     

# Define functions to solve the model
def model_setup_complete(t,k,g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na, g_DR, g_Ca, g_AHP, g_C, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        params = np.array([g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na, g_DR, g_Ca, g_AHP, g_C, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g])
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de = my_cell.dkdt()        
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()        
        dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt = my_cell.dVdt()  

        return np.array([dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, \
            dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt])

def model_setup_leak(t,k,g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        params = np.array([g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g])
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de = my_cell.dkdt()        
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()        
        dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt = my_cell.dVdt()  

        return np.array([dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, \
            dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt])

def model_setup_dynamics(t,k, g_Na, g_DR, g_Ca, g_AHP, g_C):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        params = np.array([g_Na, g_DR, g_Ca, g_AHP, g_C])
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de = my_cell.dkdt()        
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()        
        dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt = my_cell.dVdt()  

        return np.array([dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, \
            dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt])

def model_jacobian_complete(t,k,g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na, g_DR, g_Ca, g_AHP, g_C, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        params = np.array([g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na, g_DR, g_Ca, g_AHP, g_C, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g])
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        
        return my_cell.edNEG_jacobian(dense=True)

def model_jacobian_leak(t,k,g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        params = np.array([g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g])
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        
        return my_cell.edNEG_jacobian(dense=True)

def model_jacobian_dynamics(t,k,g_Na, g_DR, g_Ca, g_AHP, g_C):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        params = np.array([g_Na, g_DR, g_Ca, g_AHP, g_C])
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        
        return my_cell.edNEG_jacobian(dense=True)

def model_phi(params):     
        sol = solve_ivp(model_setup, t_span, k0, method='Radau', args=params, max_step=1e3, first_step=1e3, jac=model_jacobian_setup)
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg  = sol.y
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg = my_cell.membrane_potentials()
        return sol, phi_msn, phi_mdn, phi_msg, phi_mdg

def model_phi_last(params):     
        sol = solve_ivp(model_setup, t_span, k0, method='Radau', args=params, max_step=1e3, first_step=1e3, jac=model_jacobian_setup)
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg  = sol.y[:,-2]
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg = my_cell.membrane_potentials()
        return phi_msn, phi_mdn, phi_msg, phi_mdg, K_se, K_de

def problem_setup(setup):
        if setup == 'complete':
                problem = problem_complete
                model_setup = model_setup_complete
                model_jacobian_setup = model_jacobian_complete
                params = params_complete
                params_names = params_complete_names
                model_run = model_phi
        elif setup == 'leak':
                problem = problem_leak
                model_setup = model_setup_leak
                model_jacobian_setup = model_jacobian_leak
                params = params_leak
                params_names = params_leak_names
                model_run = model_phi
        elif setup == 'dynamics':
                problem = problem_dynamics
                model_setup = model_setup_dynamics
                model_jacobian_setup = model_jacobian_dynamics
                params = params_dynamics
                params_names = params_dynamics_names
                model_run = model_phi
        elif setup == 'grouped':
                problem = problem_complete_grouped
                model_setup = model_setup_complete
                model_jacobian_setup = model_jacobian_complete
                params = params_complete
                params_names = params_complete_names
                model_run = model_phi_last
        return problem, model_setup, model_jacobian_setup, params, params_names, model_run

# Parameters
# conductances [S/m**2]
g_Na_leak_n = 0.0246
g_K_leak_n = 0.0245
g_Cl_leak_n = 0.1   
g_Na = 30.         
g_DR = 15.           
g_Ca = 11.8          
g_AHP = .8         
g_C = 15.          
g_Na_leak_g = .1    
g_K_IR = 1.696     
g_Cl_leak_g =  0.05     

# exchanger strengths
rho_n = 1.87e-4      
U_kcc2 = 1.49e-5    
U_nkcc1 = 2.33e-5 
U_Cadec = 0.075       
rho_g = 1.12e-4  

params_complete = [g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na, g_DR, g_Ca, g_AHP, g_C, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g]
params_leak = [g_Na_leak_n, g_K_leak_n, g_Cl_leak_n, g_Na_leak_g, g_K_IR, g_Cl_leak_g, rho_n, U_kcc2, U_nkcc1, U_Cadec, rho_g]
params_dynamics = [g_Na, g_DR, g_Ca, g_AHP, g_C]

membrane_potentials_complete = ['phi_msn', 'phi_mdn', 'phi_msg', 'phi_mdg', 'K_se', 'K_de']
membrane_potentials = ['phi_msn', 'phi_msg']
params_complete_names = ['g_Na_leak_n', 'g_K_leak_n', 'g_Cl_leak_n', 'g_Na', 'g_DR', 'g_Ca', 'g_AHP', 'g_C', 'g_Na_leak_g', 'g_K_IR', 'g_Cl_leak_g', 'rho_n', 'U_kcc2', 'U_nkcc1', 'U_Cadec', 'rho_g']
params_leak_names = ['g_Na_leak_n', 'g_K_leak_n', 'g_Cl_leak_n', 'g_Na_leak_g', 'g_K_IR', 'g_Cl_leak_g', 'rho_n', 'U_kcc2', 'U_nkcc1', 'U_Cadec', 'rho_g']
params_dynamics_names = ['g_Na', 'g_DR', 'g_Ca', 'g_AHP', 'g_C']

alpha = 2
t_span = (0,12e4)

# Choose problem setup: ['complete', 'leak', 'dynamics', 'grouped']
setup = 'grouped'
problem, model_setup, model_jacobian_setup, params, params_names, model_run = problem_setup(setup)