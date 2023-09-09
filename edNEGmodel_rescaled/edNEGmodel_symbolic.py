import sympy as sp
from sympy import symbols
from sympy.physics.units import volt, second, farad, siemens, meter, mol, ampere, joule, kelvin, coulomb
from functions.initial_values_symbolic import *

""" 
    An electrodiffusive Pinsky-Rinzel model with neuron-glia interactions and cellular swelling 
    implemented in symbolical Python with sympy
"""
def nernst_potential( Z, ck_i, ck_e):
    E = R*T / (Z*F) * sp.log(ck_e / ck_i)
    return E

# t = symbols('t', positive=True)  # Time

# ions [mol]
Na_sn = symbols('Na_sn')
Na_se = symbols('Na_se')
Na_sg = symbols('Na_sg')
Na_dn = symbols('Na_dn')
Na_de = symbols('Na_de')
Na_dg = symbols('Na_dg')
K_sn = symbols('K_sn')
K_se = symbols('K_se')
K_sg = symbols('K_sg')
K_dn = symbols('K_dn')
K_de = symbols('K_de')
K_dg = symbols('K_dg')
Cl_sn = symbols('Cl_sn')
Cl_se = symbols('Cl_se')
Cl_sg = symbols('Cl_sg')
Cl_dn = symbols('Cl_dn')
Cl_de = symbols('Cl_de')
Cl_dg = symbols('Cl_dg')
Ca_sn = symbols('Ca_sn')
Ca_se = symbols('Ca_se')
Ca_dn = symbols('Ca_dn')
Ca_de = symbols('Ca_de')
n = symbols('n')
h = symbols('h')
ss = symbols('ss')
c = symbols('c')
q = symbols('q')
z = symbols('z')
V_sn = symbols('V_sn')
V_se = symbols('V_se')
V_sg = symbols('V_sg')
V_dn = symbols('V_dn')
V_de = symbols('V_de')
V_dg= symbols('V_dg')

# ion concentraions [mol/m**3]
cNa_sn = Na_sn/V_sn
cNa_se = Na_se/V_se
cNa_sg = Na_sg/V_sg
cNa_dn = Na_dn/V_dn
cNa_de = Na_de/V_de
cNa_dg = Na_dg/V_dg
cK_sn = K_sn/V_sn
cK_se = K_se/V_se
cK_sg = K_sg/V_sg
cK_dn = K_dn/V_dn
cK_de = K_de/V_de
cK_dg = K_dg/V_dg
cCl_sn = Cl_sn/V_sn
cCl_se = Cl_se/V_se 
cCl_sg = Cl_sg/V_sg 
cCl_dn = Cl_dn/V_dn 
cCl_de = Cl_de/V_de
cCl_dg = Cl_dg/V_dg
cCa_sn = Ca_sn/V_sn
cCa_se = Ca_se/V_se 
cCa_dn = Ca_dn/V_dn 
cCa_de = Ca_de/V_de
free_cCa_sn = 0.01*cCa_sn
free_cCa_dn = 0.01*cCa_dn
cX_sn = X_sn/V_sn
cX_se = X_se/V_se
cX_sg = X_sg/V_sg
cX_dn = X_dn/V_dn
cX_de = X_de/V_de
cX_dg = X_dg/V_dg

##############################################
# Define variables and parameters with units #
##############################################

# threshold concentrations for the glial Na/K pump
cNa_treshold = 10 * mol * meter**-3     # Halnes et al. 2013
cK_treshold = 1.5 * mol * meter**-3     # Halnes et al. 2013

# membrane capacitance [F/m**2]
C_msn = 3e-2 * farad * meter**-2     # Pinsky and Rinzel 1994
C_mdn = 3e-2 * farad * meter**-2     # Pinsky and Rinzel 1994
C_msg = 3e-2 * farad * meter**-2     # Pinsky and Rinzel 1994
C_mdg = 3e-2 * farad * meter**-2     # Pinsky and Rinzel 1994

# areas
alpha = 2                    # Intracellular coupling strenght
A_m = 616e-12 * meter**2     # [m**2] Saetra et al. 2020
A_i = alpha*A_m              # [m**2] Saetra et al. 2020
A_e = 61.6e-12  * meter**2   # [m**2] 
dx = 667e-6  * meter         # [m] Saetra et al. 2020

# diffusion constants [m**2/s]
D_Na = 1.33e-9 * meter**2  * second**-1  # Halnes et al. 2013
D_K = 1.96e-9  * meter**2  * second**-1  # Halnes et al. 2013 
D_Cl = 2.03e-9 * meter**2  * second**-1  # Halnes et al. 2013
D_Ca = 0.71e-9 * meter**2  * second**-1  # Halnes et al. 2016

# tortuosities
lamda_i = 3.2 # Halnes et al. 2013
lamda_e = 1.6 # Halnes et al. 2013

# valencies
Z_Na = 1.
Z_K = 1.
Z_Cl = -1.
Z_Ca = 2.
Z_X = -1.

# constants
F = 9.648e4 * coulomb * mol**-1             # [C/mol]
R = 8.314 * joule * mol**-1 * kelvin **-1   # [J/mol/K] 

# conductances [S/m**2]
g_Na_leak_n = 0.246 * siemens * meter**-2
g_K_leak_n = 0.245  * siemens * meter**-2 
g_Cl_leak_n = 1.0  * siemens * meter**-2        # Wei et al. 2014
g_Na = 300. * siemens * meter**-2               # Pinsky and Rinzel 1994
g_DR = 150. * siemens * meter**-2               # Pinsky and Rinzel 1994
g_Ca = 118. * siemens * meter**-2               # Saetra et al. 2020
g_AHP = 8.  * siemens * meter**-2               # Pinsky and Rinzel 1994
g_C = 150.  * siemens * meter**-2               # Pinsky and Rinzel 1994
g_Na_leak_g = 1.  * siemens * meter**-2         # Halnes et al. 2013
g_K_IR = 16.96  * siemens * meter**-2           # Halnes et al. 2013
g_Cl_leak_g = 0.5  * siemens * meter**-2        # Halnes et al. 2013

# exchanger strengths
rho_n = 1.87e-6 * mol * meter**-2 * second**-1      # [mol/m**2/s] Wei et al. 2014
U_kcc2 = 1.49e-7 * mol * meter**-2 * second**-1     # [mol/m**2/s] 
U_nkcc1 = 2.33e-7  * mol * meter**-2 * second**-1   # [mol/m**2/s] Wei et al. 2014
U_Cadec = 75. * second**-1                          # [1/s] Saetra et al. 2020
rho_g = 1.12e-6  * mol * meter**-2 * second**-1     # [mol/m**2/s] Halnes et al. 2013 

# water permeabilities [m**3/Pa/s] = [m**6/joule/s]
G_n = 2e-23 * meter**6 * joule**-1 * second**-1   # Dijkstra et al. 2016
G_g = 5e-23 * meter**6 * joule**-1 * second**-1   # Oestby et al. 2009

# baseline reversal potentials [V]
bE_K_sg = nernst_potential(Z_K, cbK_sg, cbK_se)
bE_K_dg = nernst_potential(Z_K, cbK_dg, cbK_de)

# solute potentials [Pa]
psi_sn = -R * T * (cNa_sn + cK_sn + cCl_sn + cCa_sn -  cM_sn)
psi_se = -R * T * (cNa_se + cK_se + cCl_se + cCa_se - cM_se)
psi_sg = -R * T * (cNa_sg + cK_sg + cCl_sg - cM_sg)
psi_dn = -R * T * (cNa_dn + cK_dn + cCl_dn + cCa_dn - cM_dn)
psi_de = -R * T * (cNa_de + cK_de + cCl_de + cCa_de - cM_de)
psi_dg = -R * T * (cNa_dg + cK_dg + cCl_dg - cM_dg)

def alpha_m(phi_m):
    phi_1 = phi_m + 0.0469 * volt
    alpha = -3.2e5 * phi_1 / (sp.exp(-phi_1 / (0.004 * volt)) - 1.) * volt**-1 * second**-1
    return alpha

def beta_m(phi_m):
    phi_2 = phi_m + 0.0199 * volt
    beta = 2.8e5 * phi_2 / (sp.exp(phi_2 / (0.005 * volt)) - 1) * volt**-1 * second**-1
    return beta

def alpha_h(phi_m):
    alpha = 128 * sp.exp((-0.043 * volt - phi_m) / (0.018*volt)) * second**-1
    return alpha

def beta_h(phi_m):
    phi_3 = phi_m + 0.02 * volt
    beta = 4000 / (1 + sp.exp(-phi_3 / (0.005*volt))) * second**-1
    return beta

def alpha_n(phi_m):
    phi_4 = phi_m + 0.0249 * volt
    alpha = - 1.6e4 * phi_4 / (sp.exp(-phi_4 / (0.005*volt)) - 1)* volt**-1 * second**-1
    return alpha

def beta_n(phi_m):
    phi_5 = phi_m + 0.04 * volt
    beta = 250 * sp.exp(-phi_5 / (0.04*volt)) * second**-1
    return beta

def alpha_s(phi_m):
    alpha = 1600 / (1 + sp.exp(-72 * (phi_m - 0.005 * volt) * volt**-1)) * second**-1
    return alpha

def beta_s(phi_m):
    phi_6 = phi_m + 0.0089 * volt
    beta = 2e4 * phi_6 / (sp.exp(phi_6 / (0.005*volt)) - 1.) * volt**-1 * second**-1
    return beta

def alpha_c(phi_m):
        phi_8 = phi_m + 50
        phi_9 = phi_m + 53.5
        if phi_m <= -10:
            alpha = 0.0527 * sp.exp(phi_8/11- phi_9/27)
        else:
            alpha = 2 * sp.exp(-phi_9 / 27)
        return alpha

def beta_c(phi_m):
    phi_9 = phi_m + 53.5
    if phi_m <= -10:
        beta = 2 * sp.exp(-phi_9 / 27) - alpha_c(phi_m)
    else:
        beta = 0.
    return beta

def chi():
    return sp.Min((free_cCa_dn-99.8e-6 * mol * meter**-3)/(2.5e-4* mol * meter**-3), 1.0)

def alpha_q():
    return sp.Min(2e4*(free_cCa_dn-99.8e-6* mol * meter**-3) * mol**-1 * meter**3 , 10.0) * second**-1

def beta_q():
    return 1.0*second**-1

def m_inf(phi_m):
    return alpha_m(phi_m) / (alpha_m(phi_m) + beta_m(phi_m))

def z_inf(phi_m):
    phi_7 = phi_m + 0.03 * volt
    return 1/(1 + sp.exp(phi_7/(0.001*volt)))

def j_pump_n(cNa_n, cK_e):
    j = (rho_n / (1.0 + sp.exp((25.* mol * meter**-3 - cNa_n)/(3.* mol * meter**-3)))) * (1.0 / (1.0 + sp.exp((3.5* mol * meter**-3 - cK_e)/(1.  * mol * meter**-3))))
    return j

def j_pump_g(cNa_g, cK_e):
    j = rho_g * (cNa_g**1.5 / (cNa_g**1.5 + cNa_treshold**1.5)) * (cK_e / (cK_e + cK_treshold))
    return j

def j_kcc2(cK_n, cK_e, cCl_n, cCl_e):
    j = U_kcc2 * sp.log(cK_n*cCl_n/(cK_e*cCl_e))
    return j

def j_nkcc1(cNa_n, cNa_e, cK_n, cK_e, cCl_n, cCl_e):
    j = U_nkcc1 * (1 / (1 + sp.exp((16* mol * meter**-3 - cK_e)/(1.  * mol * meter**-3)))) * (sp.log(cK_n*cCl_n/(cK_e*cCl_e)) + sp.log(cNa_n*cCl_n/(cNa_e*cCl_e)))
    return j

def j_Na_sn(phi_m, E_Na):
    j = g_Na_leak_n*(phi_m - E_Na) / (F*Z_Na) \
        + 3*j_pump_n(cNa_sn, cK_se) \
        + j_nkcc1(cNa_sn, cNa_se, cK_sn, cK_se, cCl_sn, cCl_se) \
        + g_Na * m_inf(phi_m)**2 * h * (phi_m - E_Na) / (F*Z_Na) \
        - 2*U_Cadec*(cCa_sn - cbCa_sn)*V_sn/A_m
    return j 

def j_K_sn(phi_m, E_K):
    j = g_K_leak_n*(phi_m - E_K) / (F*Z_K) \
        - 2*j_pump_n(cNa_sn, cK_se) \
        + j_kcc2(cK_sn, cK_se, cCl_sn, cCl_se) \
        + j_nkcc1(cNa_sn, cNa_se, cK_sn, cK_se, cCl_sn, cCl_se) \
        + g_DR * n * (phi_m - E_K) / (F*Z_K)
    return j

def j_Cl_sn(phi_m, E_Cl):
    j = g_Cl_leak_n*(phi_m - E_Cl) / (F*Z_Cl) \
        + j_kcc2(cK_sn, cK_se, cCl_sn, cCl_se) \
        + 2*j_nkcc1(cNa_sn, cNa_se, cK_sn, cK_se, cCl_sn, cCl_se)
    return j

def j_Ca_sn():
    j =  U_Cadec * (cCa_sn - cbCa_sn)*V_sn/A_m
    return j

def j_Na_dn(phi_m, E_Na):
    j = g_Na_leak_n*(phi_m - E_Na) / (F*Z_Na) \
        + 3*j_pump_n(cNa_dn, cK_de) \
        + j_nkcc1(cNa_dn, cNa_de, cK_dn, cK_de, cCl_dn, cCl_de) \
        - 2*U_Cadec*(cCa_dn - cbCa_dn)*V_dn/A_m
    return j

def j_K_dn(phi_m, E_K):
    j = g_K_leak_n*(phi_m - E_K) / (F*Z_K) \
        - 2*j_pump_n(cNa_dn, cK_de) \
        + j_kcc2(cK_dn, cK_de, cCl_dn, cCl_de) \
        + j_nkcc1(cNa_dn, cNa_de, cK_dn, cK_de, cCl_dn, cCl_de) \
        + g_AHP * q * (phi_m - E_K) / (F*Z_K) \
        + g_C * c * chi() * (phi_m - E_K) / (F*Z_K)
    return j

def j_Cl_dn(phi_m, E_Cl):
    j = g_Cl_leak_n*(phi_m - E_Cl) / (F*Z_Cl) \
        + j_kcc2(cK_dn, cK_de, cCl_dn, cCl_de) \
        + 2*j_nkcc1(cNa_dn, cNa_de, cK_dn, cK_de, cCl_dn, cCl_de)
    return j

def j_Ca_dn(phi_m, E_Ca):
    j = g_Ca * ss**2 * z * (phi_m - E_Ca) / (F*Z_Ca) \
        + U_Cadec*(cCa_dn - cbCa_dn)*V_dn/A_m
    return j

def j_Na_sg(phi_m, E_Na):
    j = g_Na_leak_g * (phi_m - E_Na) / F \
        + 3*j_pump_g(cNa_sg, cK_se)
    return j

def j_K_sg(phi_m, E_K):
    dphi = (phi_m - E_K)
    phi_m_mil = phi_m
    bE_K_mil = bE_K_sg
    fact1 = (1 + sp.exp(18.4/42.4))/(1 + sp.exp((dphi + 0.0185 * volt)/(0.0425*volt)))
    fact2 = (1 + sp.exp(-(0.1186*volt+bE_K_mil)/(0.0441*volt)))/(1+sp.exp(-(0.1186*volt+phi_m_mil)/(0.0441*volt)))
    f = sp.sqrt(cK_se/cbK_se) * fact1 * fact2 
    j = g_K_IR * f * (phi_m - E_K) / F \
        - 2 * j_pump_g(cNa_sg, cK_se)
    return j

def j_Cl_sg(phi_m, E_Cl):
    j = - g_Cl_leak_g * (phi_m - E_Cl) / F
    return j

def j_Na_dg(phi_m, E_Na):
    j = g_Na_leak_g * (phi_m - E_Na) / F \
        + 3*j_pump_g(cNa_dg, cK_de)
    return j

def j_K_dg(phi_m, E_K):
    dphi = (phi_m - E_K)
    phi_m_mil = phi_m
    bE_K_mil = bE_K_dg
    fact1 = (1 + sp.exp(18.4/42.4))/(1 + sp.exp((dphi +  0.0185*volt)/(0.0425*volt)))
    fact2 = (1 + sp.exp(-(0.1186*volt+bE_K_mil)/(0.0441*volt)))/(1+sp.exp(-(0.1186*volt+phi_m_mil)/(0.0441*volt)))
    f = sp.sqrt(cK_de/cbK_de) * fact1 * fact2 
    j = g_K_IR * f * (phi_m - E_K) / F \
        - 2 * j_pump_g(cNa_dg, cK_de)
    return j

def j_Cl_dg(phi_m, E_Cl):
    j = - g_Cl_leak_g * (phi_m - E_Cl) / F
    return j

def j_k_diff( D_k, tortuosity, ck_s, ck_d):
    j = - D_k * (ck_d - ck_s) / (tortuosity**2 * dx)
    return j

def j_k_drift( D_k, Z_k, tortuosity, ck_s, ck_d, phi_s, phi_d):
    j = - D_k * F * Z_k * (ck_d + ck_s) * (phi_d - phi_s) / (2 * tortuosity**2 * R * T * dx)
    return j

def conductivity_k( D_k, Z_k, tortuosity, ck_s, ck_d): 
    sigma = F**2 * D_k * Z_k**2 * (ck_d + ck_s) / (2 * R * T * tortuosity**2)
    return sigma

def total_charge( k):
    Z_k = [Z_Na, Z_K, Z_Cl, Z_Ca, Z_X]
    q = 0.0 * mol
    for i in range(0, 5):
        q += Z_k[i]*k[i]
    q = F*q
    return q

def reversal_potentials():
    E_Na_sn = nernst_potential(Z_Na, cNa_sn, cNa_se)
    E_Na_sg = nernst_potential(Z_Na, cNa_sg, cNa_se)
    E_Na_dn = nernst_potential(Z_Na, cNa_dn, cNa_de)
    E_Na_dg = nernst_potential(Z_Na, cNa_dg, cNa_de)
    E_K_sn = nernst_potential(Z_K, cK_sn, cK_se)
    E_K_sg = nernst_potential(Z_K, cK_sg, cK_se)
    E_K_dn = nernst_potential(Z_K, cK_dn, cK_de)
    E_K_dg = nernst_potential(Z_K, cK_dg, cK_de)
    E_Cl_sn = nernst_potential(Z_Cl, cCl_sn, cCl_se)
    E_Cl_sg = nernst_potential(Z_Cl, cCl_sg, cCl_se)
    E_Cl_dn = nernst_potential(Z_Cl, cCl_dn, cCl_de)
    E_Cl_dg = nernst_potential(Z_Cl, cCl_dg, cCl_de)
    E_Ca_sn = nernst_potential(Z_Ca, free_cCa_sn, cCa_se)
    E_Ca_dn = nernst_potential(Z_Ca, free_cCa_dn, cCa_de)
    return E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn

def membrane_potentials():
    I_n_diff = F * (Z_Na*j_k_diff(D_Na, lamda_i, cNa_sn, cNa_dn) \
        + Z_K*j_k_diff(D_K, lamda_i, cK_sn, cK_dn) \
        + Z_Cl*j_k_diff(D_Cl, lamda_i, cCl_sn, cCl_dn) \
        + Z_Ca*j_k_diff(D_Ca, lamda_i, free_cCa_sn, free_cCa_dn))
    I_g_diff = F * (Z_Na*j_k_diff(D_Na, lamda_i, cNa_sg, cNa_dg) \
        + Z_K*j_k_diff(D_K, lamda_i, cK_sg, cK_dg) \
        + Z_Cl*j_k_diff(D_Cl, lamda_i, cCl_sg, cCl_dg))
    I_e_diff = F * (Z_Na*j_k_diff(D_Na, lamda_e, cNa_se, cNa_de) \
        + Z_K*j_k_diff(D_K, lamda_e, cK_se, cK_de) \
        + Z_Cl*j_k_diff(D_Cl, lamda_e, cCl_se, cCl_de) \
        + Z_Ca*j_k_diff(D_Ca, lamda_e, cCa_se, cCa_de))

    sigma_n = conductivity_k(D_Na, Z_Na, lamda_i, cNa_sn, cNa_dn) \
        + conductivity_k(D_K, Z_K, lamda_i, cK_sn, cK_dn) \
        + conductivity_k(D_Cl, Z_Cl, lamda_i, cCl_sn, cCl_dn) \
        + conductivity_k(D_Ca, Z_Ca, lamda_i, free_cCa_sn, free_cCa_dn)
    sigma_g = conductivity_k(D_Na, Z_Na, lamda_i, cNa_sg, cNa_dg) \
        + conductivity_k(D_K, Z_K, lamda_i, cK_sg, cK_dg) \
        + conductivity_k(D_Cl, Z_Cl, lamda_i, cCl_sg, cCl_dg)
    sigma_e = conductivity_k(D_Na, Z_Na, lamda_e, cNa_se, cNa_de) \
        + conductivity_k(D_K, Z_K, lamda_e, cK_se, cK_de) \
        + conductivity_k(D_Cl, Z_Cl, lamda_e, cCl_se, cCl_de) \
        + conductivity_k(D_Ca, Z_Ca, lamda_e, cCa_se, cCa_de)

    q_dn = total_charge([Na_dn, K_dn, Cl_dn, Ca_dn, X_dn])
    q_dg = total_charge([Na_dg, K_dg, Cl_dg, 0, X_dg])
    q_sn = total_charge([Na_sn, K_sn, Cl_sn, Ca_sn, X_sn])
    q_sg = total_charge([Na_sg, K_sg, Cl_sg, 0, X_sg])

    phi_dn = q_dn / (C_mdn * A_m)
    phi_de = 0. * volt
    phi_dg = q_dg / (C_mdg * A_m)
    phi_se = ( - dx * A_i * I_n_diff + A_i * sigma_n * phi_dn - A_i * sigma_n * q_sn / (C_msn * A_m) \
        - dx * A_i * I_g_diff + A_i * sigma_g * phi_dg - A_i * sigma_g * q_sg / (C_msg * A_m) - dx * A_e * I_e_diff ) \
        / ( A_e * sigma_e + A_i * sigma_n + A_i * sigma_g )
    phi_sn = q_sn / (C_msn * A_m) + phi_se
    phi_sg = q_sg / (C_msg * A_m) + phi_se
    
    phi_msn = q_sn / (C_msn * A_m)
    phi_msg = q_sg / (C_msg * A_m)
    phi_mdn = phi_dn 
    phi_mdg = phi_dg

    return phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg

# Build current fluxes
Na_snx, Na_sex, Na_sgx, Na_dnx, Na_dex, Na_dgx, K_snx, K_sex, K_sgx, K_dnx, K_dex, K_dgx, Cl_snx, Cl_sex, Cl_sgx, Cl_dnx, Cl_dex, Cl_dgx, Ca_snx, Ca_sex, Ca_dnx, Ca_dex, nx, hx, sx, cx, qx, zx, V_snx, V_sex, V_sgx, V_dnx, V_dex, V_dgx = k0

phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg  = membrane_potentials()
E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = reversal_potentials()

j_Na_in = j_k_diff(D_Na, lamda_i, cNa_sn, cNa_dn) \
+ j_k_drift(D_Na, Z_Na, lamda_i, cNa_sn, cNa_dn, phi_sn, phi_dn) 
j_K_in = j_k_diff(D_K, lamda_i, cK_sn, cK_dn) \
    + j_k_drift(D_K, Z_K, lamda_i, cK_sn, cK_dn, phi_sn, phi_dn)
j_Cl_in = j_k_diff(D_Cl, lamda_i, cCl_sn, cCl_dn) \
    + j_k_drift(D_Cl, Z_Cl, lamda_i, cCl_sn, cCl_dn, phi_sn, phi_dn)
j_Ca_in = j_k_diff(D_Ca, lamda_i, free_cCa_sn, free_cCa_dn) \
    + j_k_drift(D_Ca, Z_Ca, lamda_i, free_cCa_sn, free_cCa_dn, phi_sn, phi_dn)

j_Na_ig = j_k_diff(D_Na, lamda_i, cNa_sg, cNa_dg) \
    + j_k_drift(D_Na, Z_Na, lamda_i, cNa_sg, cNa_dg, phi_sg, phi_dg) 
j_K_ig = j_k_diff(D_K, lamda_i, cK_sg, cK_dg) \
    + j_k_drift(D_K, Z_K, lamda_i, cK_sg, cK_dg, phi_sg, phi_dg)
j_Cl_ig = j_k_diff(D_Cl, lamda_i, cCl_sg, cCl_dg) \
    + j_k_drift(D_Cl, Z_Cl, lamda_i, cCl_sg, cCl_dg, phi_sg, phi_dg)

j_Na_e = j_k_diff(D_Na, lamda_e, cNa_se, cNa_de) \
    + j_k_drift(D_Na, Z_Na, lamda_e, cNa_se, cNa_de, phi_se, phi_de)
j_K_e = j_k_diff(D_K, lamda_e, cK_se, cK_de) \
    + j_k_drift(D_K, Z_K, lamda_e, cK_se, cK_de, phi_se, phi_de)
j_Cl_e = j_k_diff(D_Cl, lamda_e, cCl_se, cCl_de) \
    + j_k_drift(D_Cl, Z_Cl, lamda_e, cCl_se, cCl_de, phi_se, phi_de)
j_Ca_e = j_k_diff(D_Ca, lamda_e, cCa_se, cCa_de) \
    + j_k_drift(D_Ca, Z_Ca, lamda_e, cCa_se, cCa_de, phi_se, phi_de)

j_Na_msn = j_Na_sn(phi_msn, E_Na_sn)
j_K_msn = j_K_sn(phi_msn, E_K_sn)
j_Cl_msn = j_Cl_sn(phi_msn, E_Cl_sn)

j_Na_msg = j_Na_sg(phi_msg, E_Na_sg)
j_K_msg = j_K_sg(phi_msg, E_K_sg)
j_Cl_msg = j_Cl_sg(phi_msg, E_Cl_sg)

j_Na_mdn = j_Na_dn(phi_mdn, E_Na_dn)
j_K_mdn = j_K_dn(phi_mdn, E_K_dn)    
j_Cl_mdn = j_Cl_dn(phi_mdn, E_Cl_dn)

j_Na_mdg = j_Na_dg(phi_mdg, E_Na_dg)
j_K_mdg = j_K_dg(phi_mdg, E_K_dg)
j_Cl_mdg = j_Cl_dg(phi_mdg, E_Cl_dg)

j_Ca_mdn = j_Ca_dn(phi_mdn, E_Ca_dn)

dNadt_sn = -j_Na_msn*A_m - j_Na_in*A_i 
dNadt_se = j_Na_msn*A_m + j_Na_msg*A_m - j_Na_e*A_e 
dNadt_sg = -j_Na_msg*A_m - j_Na_ig*A_i
dNadt_dn = -j_Na_mdn*A_m + j_Na_in*A_i 
dNadt_de = j_Na_mdn*A_m + j_Na_mdg*A_m + j_Na_e*A_e 
dNadt_dg = -j_Na_mdg*A_m + j_Na_ig*A_i

dKdt_sn = -j_K_msn*A_m - j_K_in*A_i
dKdt_se = j_K_msn*A_m + j_K_msg*A_m - j_K_e*A_e
dKdt_sg = -j_K_msg*A_m - j_K_ig*A_i
dKdt_dn = -j_K_mdn*A_m + j_K_in*A_i
dKdt_de = j_K_mdn*A_m + j_K_mdg*A_m + j_K_e*A_e
dKdt_dg = -j_K_mdg*A_m + j_K_ig*A_i

dCldt_sn = -j_Cl_msn*A_m - j_Cl_in*A_i
dCldt_se = j_Cl_msn*A_m + j_Cl_msg*A_m - j_Cl_e*A_e
dCldt_sg = -j_Cl_msg*A_m - j_Cl_ig*A_i
dCldt_dn = -j_Cl_mdn*A_m + j_Cl_in*A_i
dCldt_de = j_Cl_mdn*A_m + j_Cl_mdg*A_m + j_Cl_e*A_e
dCldt_dg = -j_Cl_mdg*A_m + j_Cl_ig*A_i

dCadt_sn = - j_Ca_in*A_i - j_Ca_sn()*A_m
dCadt_se = - j_Ca_e*A_e + j_Ca_sn()*A_m
dCadt_dn = j_Ca_in*A_i - j_Ca_mdn*A_m 
dCadt_de = j_Ca_e*A_e + j_Ca_mdn*A_m 

dndt = alpha_n(phi_msn)*(1.0-n) - beta_n(phi_msn)*n
dhdt = alpha_h(phi_msn)*(1.0-h) - beta_h(phi_msn)*h
dsdt = alpha_s(phi_mdn)*(1.0-ss) - beta_s(phi_mdn)*ss
phi_mdn_val = phi_mdn.subs(chosen_list).subs([(Ca_dn, Ca_dnx), (Cl_dn, Cl_dnx), (K_dn,K_dnx), (Na_dn, Na_dnx)])
dcdt = alpha_c(phi_mdn_val)*(1.0-c) - beta_c(phi_mdn_val)*c
dqdt = alpha_q()*(1.0-q) - beta_q()*q
dzdt = (z_inf(phi_mdn) - z)/(1.0 * second)
dVsidt = G_n * (psi_se - psi_sn)
dVsedt = -(G_n * (psi_se - psi_sn) + G_g * (psi_se - psi_sg))
dVsgdt = G_g * (psi_se - psi_sg)
dVdidt = G_n * (psi_de - psi_dn)
dVdedt = -(G_n * (psi_de - psi_dn) + G_g * (psi_de - psi_dg))
dVdgdt = G_g * (psi_de - psi_dg)

def dkdtt(k):
    Na_snx, Na_sex, Na_sgx, Na_dnx, Na_dex, Na_dgx, K_snx, K_sex, K_sgx, K_dnx, K_dex, K_dgx, Cl_snx, Cl_sex, Cl_sgx, Cl_dnx, Cl_dex, Cl_dgx, Ca_snx, Ca_sex, Ca_dnx, Ca_dex, nx, hx, sx, cx, qx, zx, V_snx, V_sex, V_sgx, V_dnx, V_dex, V_dgx = k

    dNadt_snx = dNadt_sn.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx), (h, hx)]).subs(chosen_list)
    dNadt_sex = dNadt_se.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx), (h, hx)]).subs(chosen_list)
    dNadt_sgx = dNadt_sg.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dNadt_dnx = dNadt_dn.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dNadt_dex = dNadt_de.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dNadt_dgx = dNadt_dg.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)

    dKdt_snx = dKdt_sn.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx), (n, nx)]).subs(chosen_list)
    dKdt_sex = dKdt_se.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx), (n, nx)]).subs(chosen_list)
    dKdt_sgx = dKdt_sg.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dKdt_dnx = dKdt_dn.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx), (c, cx), (q, qx)]).subs(chosen_list)
    dKdt_dex = dKdt_de.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx), (c, cx), (q, qx)]).subs(chosen_list)
    dKdt_dgx = dKdt_dg.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)

    dCldt_snx = dCldt_sn.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dCldt_sex = dCldt_se.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dCldt_sgx = dCldt_sg.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dCldt_dnx = dCldt_dn.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dCldt_dex = dCldt_de.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dCldt_dgx = dCldt_dg.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)

    dCadt_snx = dCadt_sn.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dCadt_sex = dCadt_se.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx)]).subs(chosen_list)
    dCadt_dnx = dCadt_dn.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx), (ss, sx), (z, zx)]).subs(chosen_list)
    dCadt_dex = dCadt_de.subs([(Na_sn, Na_snx), (Na_se, Na_sex), (Na_sg, Na_sgx), (Na_dn, Na_dnx), (Na_de, Na_dex), (Na_dg, Na_dgx), \
                           (K_sn, K_snx), (K_se, K_sex), (K_sg, K_sgx), (K_dn, K_dnx), (K_de, K_dex), (K_dg, K_dgx), \
                             (Cl_sn, Cl_snx), (Cl_se, Cl_sex), (Cl_sg, Cl_sgx), (Cl_dn, Cl_dnx), (Cl_de, Cl_dex), (Cl_dg, Cl_dgx), \
                                (Ca_sn, Ca_snx), (Ca_se, Ca_sex), (Ca_dn, Ca_dnx), (Ca_de, Ca_dex), \
                                        (V_sn, V_snx), (V_se, V_sex), (V_sg, V_sgx), (V_dn, V_dnx), (V_de, V_dex), (V_dg, V_dgx), (ss, sx), (z, zx)]).subs(chosen_list)
    
    dndtx = dndt.subs([(Na_sn, Na_snx), (K_sn, K_snx), (Cl_sn, Cl_snx), (Ca_sn, Ca_snx), (n, nx)]).subs(chosen_list)
    dhdtx = dhdt.subs([(Na_sn, Na_snx), (K_sn, K_snx), (Cl_sn, Cl_snx), (Ca_sn, Ca_snx), (h, hx)]).subs(chosen_list)
    dsdtx = dsdt.subs([(Na_dn, Na_dnx), (K_dn, K_dnx), (Cl_dn, Cl_dnx), (Ca_dn, Ca_dnx), (ss, sx)]).subs(chosen_list)
    dcdtx = dcdt.subs([(Na_dn, Na_dnx), (K_dn, K_dnx), (Cl_dn, Cl_dnx), (Ca_dn, Ca_dnx), (c, cx)]).subs(chosen_list)
    dqdtx = dqdt.subs([(Ca_dn, Ca_dnx), (V_dn, V_dnx), (q, qx)]).subs(chosen_list)
    dzdtx = dzdt.subs([(Na_dn, Na_dnx), (K_dn, K_dnx), (Cl_dn, Cl_dnx), (Ca_dn, Ca_dnx), (z, zx)]).subs(chosen_list)
    dVsidtx = dVsidt.subs([(Na_sn, Na_snx), (K_sn, K_snx), (Cl_sn, Cl_snx), (Ca_sn, Ca_snx), (V_sn, V_snx), \
                        (Na_se, Na_sex), (K_se, K_sex), (Cl_se, Cl_sex), (Ca_se, Ca_sex), (V_se, V_sex)]).subs(chosen_list)
    dVsedtx = dVsedt.subs([(Na_sn, Na_snx), (K_sn, K_snx), (Cl_sn, Cl_snx), (Ca_sn, Ca_snx), (V_sn, V_snx), \
                        (Na_se, Na_sex), (K_se, K_sex), (Cl_se, Cl_sex), (Ca_se, Ca_sex), (V_se, V_sex), \
                        (Na_sg, Na_sgx), (K_sg, K_sgx), (Cl_sg, Cl_sgx), (V_sg, V_sgx)]).subs(chosen_list)
    dVsgdtx = dVsgdt.subs([(Na_se, Na_sex), (K_se, K_sex), (Cl_se, Cl_sex), (Ca_se, Ca_sex), (V_se, V_sex), \
                        (Na_sg, Na_sgx), (K_sg, K_sgx), (Cl_sg, Cl_sgx), (V_sg, V_sgx)]).subs(chosen_list)
    dVdidtx = dVdidt.subs([(Na_dn, Na_dnx), (K_dn, K_dnx), (Cl_dn, Cl_dnx), (Ca_dn, Ca_dnx), (V_dn, V_dnx), \
                        (Na_de, Na_dex), (K_de, K_dex), (Cl_de, Cl_dex), (Ca_de, Ca_dex), (V_de, V_dex)]).subs(chosen_list)
    dVdedtx = dVdedt.subs([(Na_dn, Na_dnx), (K_dn, K_dnx), (Cl_dn, Cl_dnx), (Ca_dn, Ca_dnx), (V_dn, V_dnx), \
                        (Na_de, Na_dex), (K_de, K_dex), (Cl_de, Cl_dex), (Ca_de, Ca_dex), (V_de, V_dex), \
                        (Na_dg, Na_dgx), (K_dg, K_dgx), (Cl_dg, Cl_dgx), (V_dg, V_dgx)]).subs(chosen_list)
    dVdgdtx = dVdgdt.subs([(Na_de, Na_dex), (K_de, K_dex), (Cl_de, Cl_dex), (Ca_de, Ca_dex), (V_de, V_dex), \
                        (Na_dg, Na_dgx), (K_dg, K_dgx), (Cl_dg, Cl_dgx), (V_dg, V_dgx)]).subs(chosen_list)
    
    return dNadt_snx, dNadt_sex, dNadt_sgx, dNadt_dnx, dNadt_dex, dNadt_dgx, dKdt_snx, dKdt_sex, dKdt_sgx, dKdt_dnx, dKdt_dex, dKdt_dgx, \
        dCldt_snx, dCldt_sex, dCldt_sgx, dCldt_dnx, dCldt_dex, dCldt_dgx, dCadt_snx, dCadt_sex, dCadt_dnx, dCadt_dex, \
        dndtx, dhdtx, dsdtx, dcdtx, dqdtx, dzdtx, dVsidtx, dVsedtx, dVsgdtx, dVdidtx, dVdedtx, dVdgdtx

def dkdtt_sym():
    dNadt_snx = dNadt_sn.subs(chosen_list)
    dNadt_sex = dNadt_se.subs(chosen_list)
    dNadt_sgx = dNadt_sg.subs(chosen_list)
    dNadt_dnx = dNadt_dn.subs(chosen_list)
    dNadt_dex = dNadt_de.subs(chosen_list)
    dNadt_dgx = dNadt_dg.subs(chosen_list)

    dKdt_snx = dKdt_sn.subs(chosen_list)
    dKdt_sex = dKdt_se.subs(chosen_list)
    dKdt_sgx = dKdt_sg.subs(chosen_list)
    dKdt_dnx = dKdt_dn.subs(chosen_list)
    dKdt_dex = dKdt_de.subs(chosen_list)
    dKdt_dgx = dKdt_dg.subs(chosen_list)

    dCldt_snx = dCldt_sn.subs(chosen_list)
    dCldt_sex = dCldt_se.subs(chosen_list)
    dCldt_sgx = dCldt_sg.subs(chosen_list)
    dCldt_dnx = dCldt_dn.subs(chosen_list)
    dCldt_dex = dCldt_de.subs(chosen_list)
    dCldt_dgx = dCldt_dg.subs(chosen_list)

    dCadt_snx = dCadt_sn.subs(chosen_list)
    dCadt_sex = dCadt_se.subs(chosen_list)
    dCadt_dnx = dCadt_dn.subs(chosen_list)
    dCadt_dex = dCadt_de.subs(chosen_list)
    
    dndtx = dndt.subs(chosen_list)
    dhdtx = dhdt.subs(chosen_list)
    dsdtx = dsdt.subs(chosen_list)
    dcdtx = dcdt.subs(chosen_list)
    dqdtx = dqdt.subs(chosen_list)
    dzdtx = dzdt.subs(chosen_list)
    dVsidtx = dVsidt.subs(chosen_list)
    dVsedtx = dVsedt.subs(chosen_list)
    dVsgdtx = dVsgdt.subs(chosen_list)
    dVdidtx = dVdidt.subs(chosen_list)
    dVdedtx = dVdedt.subs(chosen_list)
    dVdgdtx = dVdgdt.subs(chosen_list)
    
    return dNadt_snx, dNadt_sex, dNadt_sgx, dNadt_dnx, dNadt_dex, dNadt_dgx, dKdt_snx, dKdt_sex, dKdt_sgx, dKdt_dnx, dKdt_dex, dKdt_dgx, \
        dCldt_snx, dCldt_sex, dCldt_sgx, dCldt_dnx, dCldt_dex, dCldt_dgx, dCadt_snx, dCadt_sex, dCadt_dnx, dCadt_dex, \
        dndtx, dhdtx, dsdtx, dcdtx, dqdtx, dzdtx, dVsidtx, dVsedtx, dVsgdtx, dVdidtx, dVdedtx, dVdgdtx


# Define the ODEs
# Na_sn_ode = Eq(Na_sn.diff(t), dNadt_sn)
# Na_se_ode = Eq(Na_se.diff(t), dNadt_se)
# Na_sg_ode = Eq(Na_sg.diff(t), dNadt_sg)
# Na_dn_ode = Eq(Na_dn.diff(t), dNadt_dn)
# Na_de_ode = Eq(Na_de.diff(t), dNadt_de)
# Na_dg_ode = Eq(Na_dg.diff(t), dNadt_dg)
# K_sn_ode = Eq(K_sn.diff(t), dKdt_sn)
# K_se_ode = Eq(K_se.diff(t), dKdt_se)
# K_sg_ode = Eq(K_sg.diff(t), dKdt_sg)
# K_dn_ode = Eq(K_dn.diff(t), dKdt_dn)
# K_de_ode = Eq(K_de.diff(t), dKdt_de)
# K_dg_ode = Eq(K_dg.diff(t), dKdt_dg)
# Cl_sn_ode = Eq(Cl_sn.diff(t), dCldt_sn)
# Cl_se_ode = Eq(Cl_se.diff(t), dCldt_se)
# Cl_sg_ode = Eq(Cl_sg.diff(t), dCldt_sg)
# Cl_dn_ode = Eq(Cl_dn.diff(t), dCldt_dn)
# Cl_de_ode = Eq(Cl_de.diff(t), dCldt_de)
# Cl_dg_ode = Eq(Cl_dg.diff(t), dCldt_dg)
# Ca_sn_ode = Eq(Ca_sn.diff(t), dCadt_sn)
# Ca_se_ode = Eq(Ca_se.diff(t), dCadt_se)
# Ca_dn_ode = Eq(Ca_dn.diff(t), dCadt_dn)
# Ca_de_ode = Eq(Ca_de.diff(t), dCadt_de)
# n_ode = Eq(n.diff(t), dn)
# h_ode = Eq(h.diff(t), dh)
# s_ode = Eq(s.diff(t), ds)
# c_ode = Eq(c.diff(t), dc)
# q_ode = Eq(q.diff(t), dq)
# z_ode = Eq(z.diff(t), dz)
# V_sn_ode = Eq(V_sn.diff(t), dV_sn)
# V_se_ode = Eq(V_se.diff(t), dV_se)
# V_sg_ode = Eq(V_sg.diff(t), dV_sg)
# V_dn_ode = Eq(V_dn.diff(t), dV_dn)
# V_de_ode = Eq(V_de.diff(t), dV_de)
# V_dg_ode = Eq(V_dg.diff(t), dV_dg)

# Display the ODEs
# sp.pprint(Na_sn_ode)
# sp.pprint(K_se_ode)
# sp.pprint(n_ode)
# sp.pprint(V_sn_ode)