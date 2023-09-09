import numpy as np
import uncertainpy as un
from scipy.integrate import solve_ivp
from edNEGmodel_parameters import *
from functions.initial_values_transposed import *
import chaospy as cp

def model_setup(t,k, g_Na, g_DR, g_Ca, g_AHP, g_C):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        params = np.array([g_Na, g_DR, g_Ca, g_AHP, g_C])
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de = my_cell.dkdt()        
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()        
        dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt = my_cell.dVdt()  

        if t > stim_start and t < stim_end:
            dKdt_sn += I_stim / my_cell.F
            dKdt_se -= I_stim / my_cell.F

        return np.array([dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, \
            dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt])


def model_jacobian_setup(t,k,g_Na, g_DR, g_Ca, g_AHP, g_C):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        params = np.array([g_Na, g_DR, g_Ca, g_AHP, g_C])
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        
        return my_cell.edNEG_jacobian(dense=True)

def phi_msn(g_Na = 30.  ,       
            g_DR = 15.   ,        
            g_Ca = 11.8  ,        
            g_AHP = .8   ,      
            g_C = 15. ):  
        
        params = [g_Na, g_DR, g_Ca, g_AHP, g_C]
        sol = solve_ivp(model_setup, t_span, k0, method='Radau', args=params, max_step=1e3, t_eval=tt, jac=model_jacobian_setup)
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg  = sol.y
        my_cell = edNEGmodel(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
        phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg = my_cell.membrane_potentials()
        
        time = sol.t
        values = phi_msn
        info = {"stimulus_start": stim_start, "stimulus_end": stim_end} #, "K_se" : K_se}

        return time, values, info

# def K_se(time, values, info):
#     return time, info["K_se"]

# Transposed test parameters
I_stim = 8e-5 # [A]
alpha = 2
t_dur = 3e3       # [s]
t_span = (0, t_dur)
stim_start = 0.2e3
stim_end = 2.8e3 

par = [30. , 15. , 11.8 , .8 , 15.]
sol = solve_ivp(model_setup, t_span, k0, method='Radau', args=par, max_step=1e3, jac=model_jacobian_setup)
tt = sol.t

if __name__ == '__main__': 

    # Initialize the model
    model = un.Model(run=phi_msn,
                    labels=["Time (ms)", "Membrane potential (mV)"])
    
    # # Add second output
    # feature_list = [K_se]
    
    # Initialize the features 
    feature_list = un.SpikingFeatures(features_to_run="all")  

    # Define a parameter dictionary
    parameters = {  "g_Na" : 30.  ,  # [mS cm**-2]
                    "g_DR" : 15. ,  # [mS cm**-2]
                    "g_Ca" : 11.8,    # membrane capacitance [uF cm**-2]
                    "g_AHP" : .8  , # [mV]
                    "g_C" : 15.
                    }

    # Create the parameters
    parameters = un.Parameters(parameters)

    # Set all parameters to have a uniform distribution
    # within a 10% interval around their fixed value
    parameters.set_all_distributions(un.uniform(0.2))

    # Perform the uncertainty quantification
    UQ = un.UncertaintyQuantification(model,
                                    parameters=parameters,
                                    features=feature_list)
    
    # We set the seed to easier be able to reproduce the result
    data = UQ.quantify(seed=10, rosenblatt=True ,data_folder = "dynamic/phi_msn_10/data")

    # Plot first total Sobol' indices
    un.plotting.PlotUncertainty(filename=u"dynamic/phi_msn_10/data/phi_msn.h5", folder=u'dynamic/phi_msn_10/sobol_total/', figureformat=u'.eps', logger_level=u'info').plot_condensed(sensitivity="sobol_total")
