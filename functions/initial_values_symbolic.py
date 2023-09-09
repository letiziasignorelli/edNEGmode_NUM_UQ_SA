import numpy as np
import pkg_resources
from sympy.physics.units import volt, second, farad, siemens, meter, mol, ampere, joule, kelvin, coulomb

centi = 1e2
milli = 1e3
micro = 1e6
nano = 1e9
pico = 1e12
conversion_list = [(volt,milli), (second, milli), (farad, micro), \
                    (siemens, milli), (meter, centi), (mol, nano), \
                        (ampere, micro), (joule, pico), (kelvin,1), (coulomb, nano)]
substitution_list = [(volt,1), (second, 1), (farad, 1), \
                    (siemens, 1), (meter, 1), (mol, 1), \
                        (ampere, 1), (joule, 1), (kelvin,1), (coulomb, 1)]

chosen_list = conversion_list

filename = pkg_resources.resource_filename('data', 'initial_values/initial_values.npz')
data = np.load(filename)
 
T = 309.14 * kelvin
 
# initial membrane potentials [V]
phi_msn0 = data['phi_msn'] * volt
phi_msg0 = data['phi_msg'] * volt
phi_mdn0 = data['phi_mdn'] * volt
phi_mdg0 = data['phi_mdg'] * volt

# initial volumes [m**3]
V_sn0 = data['V_sn'] * meter**3
V_se0 = data['V_se'] * meter**3
V_sg0 = data['V_sg'] * meter**3
V_dn0 = data['V_dn'] * meter**3
V_de0 = data['V_de'] * meter**3
V_dg0 = data['V_dg'] * meter**3

# initial amounts of ions [mol]
Na_sn0 = data['Na_sn'] * mol
Na_se0 = data['Na_se'] * mol
Na_sg0 = data['Na_sg'] * mol
K_sn0 = data['K_sn'] * mol
K_se0 = data['K_se'] * mol
K_sg0 = data['K_sg'] * mol
Cl_sn0 = data['Cl_sn'] * mol
Cl_se0 = data['Cl_se'] * mol
Cl_sg0 = data['Cl_sg'] * mol
Ca_sn0 = data['Ca_sn'] * mol
Ca_se0 = data['Ca_se'] * mol

Na_dn0 = data['Na_dn'] * mol
Na_de0 = data['Na_de'] * mol
Na_dg0 = data['Na_dg'] * mol
K_dn0 = data['K_dn']  * mol
K_de0 = data['K_de'] * mol
K_dg0 = data['K_dg'] * mol
Cl_dn0 = data['Cl_dn'] * mol
Cl_de0 = data['Cl_de'] * mol
Cl_dg0 = data['Cl_dg'] * mol
Ca_dn0 = data['Ca_dn'] * mol
Ca_de0 = data['Ca_de'] * mol

# intial gating variables
n0 = data['n'].item()
h0 = data['h'].item()
s0 = data['s'].item()
c0 = data['c'].item()
q0 = data['q'].item()
z0 = data['z'].item()

# baseline ion concentrations [mol/m**3]
cbK_se = 3.082 * mol * meter**-3
cbK_sg = 99.959 * mol * meter**-3
cbK_de = 3.082 * mol * meter**-3
cbK_dg = 99.959 * mol * meter**-3
cbCa_sn = 0.01 * mol * meter**-3
cbCa_dn = 0.01 * mol * meter**-3

# residual charges [mol]
res_sn = phi_msn0*3e-2*616e-12/9.648e4 * mol * volt**-1
res_sg = phi_msg0*3e-2*616e-12/9.648e4 * mol * volt**-1
res_se = res_sn+res_sg
res_dn = phi_mdn0*3e-2*616e-12/9.648e4 * mol * volt**-1
res_dg = phi_mdg0*3e-2*616e-12/9.648e4 * mol * volt**-1
res_de = res_dn+res_dg

X_sn = Na_sn0 + K_sn0 - Cl_sn0 + 2*Ca_sn0 - res_sn
X_se = Na_se0 + K_se0 - Cl_se0 + 2*Ca_se0 + res_se
X_sg = Na_sg0 + K_sg0 - Cl_sg0 - res_sg
X_dn = Na_dn0 + K_dn0 - Cl_dn0 + 2*Ca_dn0 - res_dn
X_de = Na_de0 + K_de0 - Cl_de0 + 2*Ca_de0 + res_de
X_dg = Na_dg0 + K_dg0 - Cl_dg0 - res_dg

# residual mass [mol/m**3]
cM_sn = (Na_sn0 + K_sn0 + Cl_sn0 + Ca_sn0)/V_sn0
cM_se = (Na_se0 + K_se0 + Cl_se0 + Ca_se0)/V_se0 
cM_sg = (Na_sg0 + K_sg0 + Cl_sg0)/V_sg0
cM_dn = (Na_dn0 + K_dn0 + Cl_dn0 + Ca_dn0)/V_dn0
cM_de = (Na_de0 + K_de0 + Cl_de0 + Ca_de0)/V_de0 
cM_dg = (Na_dg0 + K_dg0 + Cl_dg0)/V_dg0 

k0 = [Na_sn0.subs(chosen_list), Na_se0.subs(chosen_list), Na_sg0.subs(chosen_list), Na_dn0.subs(chosen_list), Na_de0.subs(chosen_list), Na_dg0.subs(chosen_list),\
      K_sn0.subs(chosen_list), K_se0.subs(chosen_list), K_sg0.subs(chosen_list), K_dn0.subs(chosen_list), K_de0.subs(chosen_list), K_dg0.subs(chosen_list),\
      Cl_sn0.subs(chosen_list), Cl_se0.subs(chosen_list), Cl_sg0.subs(chosen_list), Cl_dn0.subs(chosen_list), Cl_de0.subs(chosen_list), Cl_dg0.subs(chosen_list), \
      Ca_sn0.subs(chosen_list), Ca_se0.subs(chosen_list), Ca_dn0.subs(chosen_list), Ca_de0.subs(chosen_list), \
      n0, h0, s0, c0, q0, z0,\
      V_sn0.subs(chosen_list), V_se0.subs(chosen_list), V_sg0.subs(chosen_list), V_dn0.subs(chosen_list), V_de0.subs(chosen_list), V_dg0.subs(chosen_list)]

