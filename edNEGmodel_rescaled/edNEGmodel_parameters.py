import numpy as np
from scipy.sparse import csr_matrix

class edNEGmodel():
    """ 
    An electrodiffusive Pinsky-Rinzel model with neuron-glia interactions and cellular swelling 
    with units rescaled and selecting uncertain parameters
    
    Methods
    -------
    constructor(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, \
        Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, \
        X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, \
        cbK_se, cbK_sg, cbK_de, cbK_dg, \
        cbCa_sn, cbCa_dn, n, h, s, c, q, z, \
        V_sn, V_se, V_sg, V_dn, V_de, V_dg, \
        cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params)
    j_Na_sn(phi_m, E_Na): compute the Na+ membrane flux (soma layer, neuron)
    j_K_sn(phi_m, E_K): compute the K+ membrane flux (soma layer, neuron)
    j_Cl_sn(phi_m, E_Cl): compute the Cl- membrane flux (soma layer, neuron) 
    j_Ca_sn(): compute the Ca2+ membrane flux (soma layer, neuron) 
    j_Na_dn(phi_m, E_Na): compute the Na+ membrane flux (dendrite layer, neuron)
    j_K_dn(phi_m, E_K): compute the K+ membrane flux (dendrite layer, neuron)
    j_Cl_dn(phi_m, E_Cl): compute the Cl- membrane flux (dendrite layer, neuron) 
    j_Ca_dn(phi_m, E_Cl): compute the Ca2+ membrane flux (dendrite layer, neuron) 
    j_Na_sg(phi_m, E_Na): compute the Na+ membrane flux (soma layer, glia)
    j_K_sg(phi_m, E_K): compute the K+ membrane flux (soma layer, glia)
    j_Cl_sg(phi_m, E_Cl): compute the Cl- membrane flux (soma layer, glia) 
    j_Na_dg(phi_m, E_Na): compute the Na+ membrane flux (dendrite layer, glia)
    j_K_dg(phi_m, E_K): compute the K+ membrane flux (dendrite layer, glia)
    j_Cl_dg(phi_m, E_Cl): compute the Cl- membrane flux (dendrite layer, glia) 
    j_pump_n(cNa_n, cK_e): compute the Na+/K+ pump flux across neuronal membrane
    j_pump_g(cNa_g, cK_e): compute the Na+/K+ pump flux across glial membrane
    j_kcc2(cK_n, cK_e, cCl_n, cCl_e): compute the K+/Cl- cotransporter flux across neuronal membrane
    j_nkcc1(cNa_n, cNa_e, cK_n, cK_e, cCl_n, cCl_e): compute the Na+/K+/Cl- cotransporter flux across neuronal membrane
    j_k_diff(D_k, tortuosity, ck_s, ck_d): compute the axial diffusion flux of ion k
    j_k_drift(D_k, Z_k, tortuosity, ck_s, ck_d, phi_s, phi_d): compute the axial drift flux of ion k
    conductivity_k(D_k, Z_k, tortuosity, ck_s, ck_d): compute axial conductivity of ion k
    total_charge(k): calculate the total charge within a compartment
    nernst_potential(Z, ck_i, ck_e): calculate the reversal potential of ion k
    reversal_potentials(): calculate the reversal potentials of all ion species
    membrane_potentials(): calculate the membrane potentials
    dkdt(): calculate dk/dt for all ion species k
    dmdt(): calculate dm/dt for all gating particles m
    dVdt(): calculate dV/dt for all volumes V
    d... : compute derivatives of every function
    ...
    edNEG_jacobian(dense): compute and assemble Jacobian matrix as a dense matrix if dense=True (default=False)
    """


    def __init__(self, T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, \
        Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, \
        X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, \
        cbK_se, cbK_sg, cbK_de, cbK_dg, \
        cbCa_sn, cbCa_dn, n, h, s, c, q, z, \
        V_sn, V_se, V_sg, V_dn, V_de, V_dg, \
        cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg, params):

        # temperature [K]
        self.T = T

        # ions [mol]
        self.Na_sn = Na_sn
        self.Na_se = Na_se
        self.Na_sg = Na_sg
        self.Na_dn = Na_dn
        self.Na_de = Na_de
        self.Na_dg = Na_dg
        self.K_sn = K_sn
        self.K_se = K_se
        self.K_sg = K_sg
        self.K_dn = K_dn
        self.K_de = K_de
        self.K_dg = K_dg
        self.Cl_sn = Cl_sn
        self.Cl_se = Cl_se 
        self.Cl_sg = Cl_sg 
        self.Cl_dn = Cl_dn 
        self.Cl_de = Cl_de
        self.Cl_dg = Cl_dg
        self.Ca_sn = Ca_sn
        self.Ca_se = Ca_se 
        self.Ca_dn = Ca_dn 
        self.Ca_de = Ca_de
        self.X_sn = X_sn
        self.X_se = X_se
        self.X_sg = X_sg
        self.X_dn = X_dn
        self.X_de = X_de
        self.X_dg = X_dg
        
        # ion concentraions [mol/m**3]
        self.cNa_sn = Na_sn/V_sn
        self.cNa_se = Na_se/V_se
        self.cNa_sg = Na_sg/V_sg
        self.cNa_dn = Na_dn/V_dn
        self.cNa_de = Na_de/V_de
        self.cNa_dg = Na_dg/V_dg
        self.cK_sn = K_sn/V_sn
        self.cK_se = K_se/V_se
        self.cK_sg = K_sg/V_sg
        self.cK_dn = K_dn/V_dn
        self.cK_de = K_de/V_de
        self.cK_dg = K_dg/V_dg
        self.cCl_sn = Cl_sn/V_sn
        self.cCl_se = Cl_se/V_se 
        self.cCl_sg = Cl_sg/V_sg 
        self.cCl_dn = Cl_dn/V_dn 
        self.cCl_de = Cl_de/V_de
        self.cCl_dg = Cl_dg/V_dg
        self.cCa_sn = Ca_sn/V_sn
        self.cCa_se = Ca_se/V_se 
        self.cCa_dn = Ca_dn/V_dn 
        self.cCa_de = Ca_de/V_de
        self.free_cCa_sn = 0.01*self.cCa_sn
        self.free_cCa_dn = 0.01*self.cCa_dn
        self.cX_sn = X_sn/V_sn
        self.cX_se = X_se/V_se
        self.cX_sg = X_sg/V_sg
        self.cX_dn = X_dn/V_dn
        self.cX_de = X_de/V_de
        self.cX_dg = X_dg/V_dg

        # concentrations of static molecules without charge [mol*m**-3] 
        self.cM_sn = cM_sn
        self.cM_se = cM_se
        self.cM_sg = cM_sg
        self.cM_dn = cM_dn
        self.cM_de = cM_de
        self.cM_dg = cM_dg

        # gating variables
        self.n = n
        self.h = h
        self.s = s
        self.c = c
        self.q = q
        self.z = z
        
        # baseline concentrations [mol/m**3] 
        self.cbK_se = cbK_se           
        self.cbK_sg = cbK_sg          
        self.cbK_de = cbK_de     
        self.cbK_dg = cbK_dg
        self.cbCa_sn = cbCa_sn
        self.cbCa_dn = cbCa_dn

        # threshold concentrations for the glial Na/K pump
        self.cNa_treshold = 10000 # Halnes et al. 2013
        self.cK_treshold = 1500 # Halnes et al. 2013

        # membrane capacitance [F/m**2]
        self.C_msn = 3 # Pinsky and Rinzel 1994
        self.C_mdn = 3 # Pinsky and Rinzel 1994
        self.C_msg = 3 # Pinsky and Rinzel 1994
        self.C_mdg = 3 # Pinsky and Rinzel 1994
       
        # volumes and areas
        self.alpha = alpha
        self.A_m = 616e-8               # [m**2] Saetra et al. 2020
        self.A_i = self.alpha*self.A_m    # [m**2] Saetra et al. 2020
        self.A_e = 61.6e-8               # [m**2] 
        self.dx = 667e-4                  # [m] Saetra et al. 2020 self.V_sn = V_sn                  
        self.V_sn = V_sn                  # [m**3]
        self.V_se = V_se                  # [m**3]
        self.V_sg = V_sg                  # [m**3]
        self.V_dn = V_dn                  # [m**3]
        self.V_de = V_de                  # [m**3]
        self.V_dg = V_dg                  # [m**3]
 
        # diffusion constants [m**2/s]
        self.D_Na = 1.33e-8 # Halnes et al. 2013
        self.D_K = 1.96e-8  # Halnes et al. 2013 
        self.D_Cl = 2.03e-8 # Halnes et al. 2013
        self.D_Ca = 0.71e-8 # Halnes et al. 2016

        # tortuosities
        self.lamda_i = 3.2 # Halnes et al. 2013
        self.lamda_e = 1.6 # Halnes et al. 2013

        # valencies
        self.Z_Na = 1.
        self.Z_K = 1.
        self.Z_Cl = -1.
        self.Z_Ca = 2.
        self.Z_X = -1.

        # constants
        self.F = 96480    # [C/mol]
        self.R = 8314      # [J/mol/K] 


        ########################
        # Uncertain parameters #
        ########################

        # All selected uncertain parameters
        if len(params) == 16 :
            self.g_Na_leak_n = params[0]
            self.g_K_leak_n = params[1]  
            self.g_Cl_leak_n = params[2]        
            self.g_Na = params[3]               
            self.g_DR = params[4]               
            self.g_Ca = params[5]               
            self.g_AHP = params[6]              
            self.g_C = params[7]                
            self.g_Na_leak_g = params[8]        
            self.g_K_IR = params[9]             
            self.g_Cl_leak_g = params[10]       
            
            # exchanger strengths
            self.rho_n = params[11]             
            self.U_kcc2 = params[12]             
            self.U_nkcc1 = params[13]           
            self.U_Cadec = params[14]           
            self.rho_g = params[15]             
        
        # Only uncertain parameters affecting the dynamical steady state
        elif len(params) == 5 :
            self.g_Na = params[0]               
            self.g_DR = params[1]               
            self.g_Ca = params[2]               
            self.g_AHP = params[3]              
            self.g_C = params[4]                

            self.g_Na_leak_n = 0.0246
            self.g_K_leak_n = 0.0245
            self.g_Cl_leak_n = 0.1  
            self.g_Na_leak_g = .1    
            self.g_K_IR = 1.696     
            self.g_Cl_leak_g =  0.05     
            self.rho_n = 1.87e-4      
            self.U_kcc2 = 1.49e-5    
            self.U_nkcc1 = 2.33e-5 
            self.U_Cadec = 0.075       
            self.rho_g = 1.12e-4

        # Only uncertain parameters affecting the resting
        if len(params) == 11 :
            self.g_Na_leak_n = params[0]
            self.g_K_leak_n = params[1]  
            self.g_Cl_leak_n = params[2]        
            self.g_Na_leak_g = params[3]        
            self.g_K_IR = params[4]             
            self.g_Cl_leak_g = params[5]        
            self.rho_n = params[6]              
            self.U_kcc2 = params[7]              
            self.U_nkcc1 = params[8]            
            self.U_Cadec = params[9]            
            self.rho_g = params[10]             

            self.g_Na = 30.         
            self.g_DR = 15.           
            self.g_Ca = 11.8          
            self.g_AHP = .8         
            self.g_C = 15. 
        
        # water permeabilities [m**3/Pa/s] 
        self.G_n = 2e-26    # Dijkstra et al. 2016
        self.G_g = 5e-26    # Oestby et al. 2009
        
        # baseline reversal potentials [V]
        self.bE_K_sg = self.nernst_potential(self.Z_K, self.cbK_sg, self.cbK_se)
        self.bE_K_dg = self.nernst_potential(self.Z_K, self.cbK_dg, self.cbK_de)

        # solute potentials [Pa]
        self.psi_sn = -self.R * self.T * (self.cNa_sn + self.cK_sn + self.cCl_sn + self.cCa_sn -  cM_sn)
        self.psi_se = -self.R * self.T * (self.cNa_se + self.cK_se + self.cCl_se + self.cCa_se - cM_se)
        self.psi_sg = -self.R * self.T * (self.cNa_sg + self.cK_sg + self.cCl_sg - cM_sg)
        self.psi_dn = -self.R * self.T * (self.cNa_dn + self.cK_dn + self.cCl_dn + self.cCa_dn - cM_dn)
        self.psi_de = -self.R * self.T * (self.cNa_de + self.cK_de + self.cCl_de + self.cCa_de - cM_de)
        self.psi_dg = -self.R * self.T * (self.cNa_dg + self.cK_dg + self.cCl_dg - cM_dg)

    def alpha_m(self, phi_m):
        phi_1 = phi_m + 46.9
        alpha = -0.32 * phi_1 / (np.exp(-phi_1 / 4) - 1.)
        return alpha

    def beta_m(self, phi_m):
        phi_2 = phi_m + 19.9
        beta = 0.28 * phi_2 / (np.exp(phi_2 / 5) - 1)
        return beta

    def alpha_h(self, phi_m):
        alpha = 0.128 * np.exp((-43 - phi_m) / 18)
        return alpha

    def beta_h(self, phi_m):
        phi_3 = phi_m + 20
        beta = 4 / (1 + np.exp(-phi_3 / 5))
        return beta

    def alpha_n(self, phi_m):
        phi_4 = phi_m + 24.9
        alpha = - 0.016 * phi_4 / (np.exp(-phi_4 / 5) - 1)
        return alpha

    def beta_n(self, phi_m):
        phi_5 = phi_m + 40
        beta = 0.25 * np.exp(-phi_5 / 40)
        return beta

    def alpha_s(self, phi_m):
        alpha = 1.6 / (1 + np.exp(-0.072 * (phi_m - 5)))
        return alpha

    def beta_s(self, phi_m):
        phi_6 = phi_m + 8.9
        beta = 0.02 * phi_6 / (np.exp(phi_6 / 5) - 1.)
        return beta

    def alpha_c(self, phi_m):
        phi_8 = phi_m + 50
        phi_9 = phi_m + 53.5
        if phi_m <= -10:
            alpha = 0.0527 * np.exp(phi_8/11- phi_9/27)
        else:
            alpha = 2 * np.exp(-phi_9 / 27)
        return alpha

    def beta_c(self, phi_m):
        phi_9 = phi_m + 53.5
        if phi_m <= -10:
            beta = 2 * np.exp(-phi_9 / 27) - self.alpha_c(phi_m)
        else:
            beta = 0.
        return beta
    
    def chi(self):
        return min((self.free_cCa_dn-99.8e-3)/2.5e-1, 1.0)

    def alpha_q(self):
        return min(0.02*(self.free_cCa_dn-99.8e-3), 0.01) 

    def beta_q(self):
        return 0.001
    
    def z_inf(self, phi_m):
        phi_7 = phi_m + 30
        return 1/(1 + np.exp(phi_7/1.))
    
    def m_inf(self, phi_m):
        return self.alpha_m(phi_m) / (self.alpha_m(phi_m) + self.beta_m(phi_m))

    def j_pump_n(self, cNa_n, cK_e):
        j = (self.rho_n / (1.0 + np.exp((25000. - cNa_n)/3000.))) * (1.0 / (1.0 + np.exp((3500 - cK_e)/1000)))
        return j

    def j_pump_g(self, cNa_g, cK_e):
        j = self.rho_g * (cNa_g**1.5 / (cNa_g**1.5 + self.cNa_treshold**1.5)) * (cK_e / (cK_e + self.cK_treshold))
        return j

    def j_kcc2(self, cK_n, cK_e, cCl_n, cCl_e):
        j = self.U_kcc2 * np.log(cK_n*cCl_n/(cK_e*cCl_e))
        return j
    
    def j_nkcc1(self, cNa_n, cNa_e, cK_n, cK_e, cCl_n, cCl_e):
        j = self.U_nkcc1 * (1 / (1 + np.exp((16e3 - cK_e)/1e3))) * (np.log(cK_n*cCl_n/(cK_e*cCl_e)) + np.log(cNa_n*cCl_n/(cNa_e*cCl_e)))
        return j
 
    def j_Na_sn(self, phi_m, E_Na):
        j = self.g_Na_leak_n*(phi_m - E_Na) / (self.F*self.Z_Na) \
            + 3*self.j_pump_n(self.cNa_sn, self.cK_se) \
            + self.j_nkcc1(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se) \
            + self.g_Na * self.m_inf(phi_m)**2 * self.h * (phi_m - E_Na) / (self.F*self.Z_Na) \
            - 2*self.U_Cadec*(self.cCa_sn - self.cbCa_sn)*self.V_sn/self.A_m
        return j 

    def j_K_sn(self, phi_m, E_K):
        j = self.g_K_leak_n*(phi_m - E_K) / (self.F*self.Z_K) \
            - 2*self.j_pump_n(self.cNa_sn, self.cK_se) \
            + self.j_kcc2(self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se) \
            + self.j_nkcc1(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se) \
            + self.g_DR * self.n * (phi_m - E_K) / (self.F*self.Z_K)
        return j

    def j_Cl_sn(self, phi_m, E_Cl):
        j = self.g_Cl_leak_n*(phi_m - E_Cl) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se) \
            + 2*self.j_nkcc1(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se)
        return j

    def j_Ca_sn(self):
        j =  self.U_Cadec * (self.cCa_sn - self.cbCa_sn)*self.V_sn/self.A_m
        return j

    def j_Na_dn(self, phi_m, E_Na):
        j = self.g_Na_leak_n*(phi_m - E_Na) / (self.F*self.Z_Na) \
            + 3*self.j_pump_n(self.cNa_dn, self.cK_de) \
            + self.j_nkcc1(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de) \
            - 2*self.U_Cadec*(self.cCa_dn - self.cbCa_dn)*self.V_dn/self.A_m
        return j

    def j_K_dn(self, phi_m, E_K):
        j = self.g_K_leak_n*(phi_m - E_K) / (self.F*self.Z_K) \
            - 2*self.j_pump_n(self.cNa_dn, self.cK_de) \
            + self.j_kcc2(self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de) \
            + self.j_nkcc1(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de) \
            + self.g_AHP * self.q * (phi_m - E_K) / (self.F*self.Z_K) \
            + self.g_C * self.c * self.chi() * (phi_m - E_K) / (self.F*self.Z_K)
        return j

    def j_Cl_dn(self, phi_m, E_Cl):
        j = self.g_Cl_leak_n*(phi_m - E_Cl) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de) \
            + 2*self.j_nkcc1(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de)
        return j

    def j_Ca_dn(self, phi_m, E_Ca):
        j = self.g_Ca * self.s**2 * self.z * (phi_m - E_Ca) / (self.F*self.Z_Ca) \
            + self.U_Cadec*(self.cCa_dn - self.cbCa_dn)*self.V_dn/self.A_m
        return j

    def j_Na_sg(self, phi_m, E_Na):
        j = self.g_Na_leak_g * (phi_m - E_Na) / self.F \
            + 3*self.j_pump_g(self.cNa_sg, self.cK_se)
        return j

    def j_K_sg(self, phi_m, E_K):
        dphi = (phi_m - E_K)
        phi_m_mil = phi_m
        bE_K_mil = self.bE_K_sg
        fact1 = (1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))
        fact2 = (1 + np.exp(-(118.6+bE_K_mil)/44.1))/(1+np.exp(-(118.6+phi_m_mil)/44.1))
        f = np.sqrt(self.cK_se/self.cbK_se) * fact1 * fact2 
        j = self.g_K_IR * f * (phi_m - E_K) / self.F \
            - 2 * self.j_pump_g(self.cNa_sg, self.cK_se)
        return j

    def j_Cl_sg(self, phi_m, E_Cl):
        j = - self.g_Cl_leak_g * (phi_m - E_Cl) / self.F
        return j

    def j_Na_dg(self, phi_m, E_Na):
        j = self.g_Na_leak_g * (phi_m - E_Na) / self.F \
            + 3*self.j_pump_g(self.cNa_dg, self.cK_de)
        return j

    def j_K_dg(self, phi_m, E_K):
        dphi = (phi_m - E_K)
        phi_m_mil = phi_m
        bE_K_mil = self.bE_K_dg
        fact1 = (1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))
        fact2 = (1 + np.exp(-(118.6+bE_K_mil)/44.1))/(1+np.exp(-(118.6+phi_m_mil)/44.1))
        f = np.sqrt(self.cK_de/self.cbK_de) * fact1 * fact2 
        j = self.g_K_IR * f * (phi_m - E_K) / self.F \
            - 2 * self.j_pump_g(self.cNa_dg, self.cK_de)
        return j

    def j_Cl_dg(self, phi_m, E_Cl):
        j = - self.g_Cl_leak_g * (phi_m - E_Cl) / self.F
        return j

    def j_k_diff(self, D_k, tortuosity, ck_s, ck_d):
        j = - D_k * (ck_d - ck_s) / (tortuosity**2 * self.dx)
        return j

    def j_k_drift(self, D_k, Z_k, tortuosity, ck_s, ck_d, phi_s, phi_d):
        j = - D_k * self.F * Z_k * (ck_d + ck_s) * (phi_d - phi_s) / (2 * tortuosity**2 * self.R * self.T * self.dx)
        return j

    def conductivity_k(self, D_k, Z_k, tortuosity, ck_s, ck_d): 
        sigma = self.F**2 * D_k * Z_k**2 * (ck_d + ck_s) / (2 * self.R * self.T * tortuosity**2)
        return sigma

    def total_charge(self, k):
        Z_k = [self.Z_Na, self.Z_K, self.Z_Cl, self.Z_Ca, self.Z_X]
        q = 0.0
        for i in range(0, 5):
            q += Z_k[i]*k[i]
        q = self.F*q
        return q

    def nernst_potential(self, Z, ck_i, ck_e):
        E = self.R*self.T / (Z*self.F) * np.log(ck_e / ck_i)
        return E

    def reversal_potentials(self):
        E_Na_sn = self.nernst_potential(self.Z_Na, self.cNa_sn, self.cNa_se)
        E_Na_sg = self.nernst_potential(self.Z_Na, self.cNa_sg, self.cNa_se)
        E_Na_dn = self.nernst_potential(self.Z_Na, self.cNa_dn, self.cNa_de)
        E_Na_dg = self.nernst_potential(self.Z_Na, self.cNa_dg, self.cNa_de)
        E_K_sn = self.nernst_potential(self.Z_K, self.cK_sn, self.cK_se)
        E_K_sg = self.nernst_potential(self.Z_K, self.cK_sg, self.cK_se)
        E_K_dn = self.nernst_potential(self.Z_K, self.cK_dn, self.cK_de)
        E_K_dg = self.nernst_potential(self.Z_K, self.cK_dg, self.cK_de)
        E_Cl_sn = self.nernst_potential(self.Z_Cl, self.cCl_sn, self.cCl_se)
        E_Cl_sg = self.nernst_potential(self.Z_Cl, self.cCl_sg, self.cCl_se)
        E_Cl_dn = self.nernst_potential(self.Z_Cl, self.cCl_dn, self.cCl_de)
        E_Cl_dg = self.nernst_potential(self.Z_Cl, self.cCl_dg, self.cCl_de)
        E_Ca_sn = self.nernst_potential(self.Z_Ca, self.free_cCa_sn, self.cCa_se)
        E_Ca_dn = self.nernst_potential(self.Z_Ca, self.free_cCa_dn, self.cCa_de)
        return E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn

    def membrane_potentials(self, extra=False):
        I_n_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_sn, self.cNa_dn) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.cK_sn, self.cK_dn) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn))
        I_g_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_sg, self.cNa_dg) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.cK_sg, self.cK_dg) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg))
        I_e_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_e, self.cNa_se, self.cNa_de) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_e, self.cK_se, self.cK_de) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_e, self.cCl_se, self.cCl_de) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_e, self.cCa_se, self.cCa_de))

        sigma_n = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn)
        sigma_g = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg)
        sigma_e = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de)

        q_dn = self.total_charge([self.Na_dn, self.K_dn, self.Cl_dn, self.Ca_dn, self.X_dn])
        q_dg = self.total_charge([self.Na_dg, self.K_dg, self.Cl_dg, 0, self.X_dg])
        q_sn = self.total_charge([self.Na_sn, self.K_sn, self.Cl_sn, self.Ca_sn, self.X_sn])
        q_sg = self.total_charge([self.Na_sg, self.K_sg, self.Cl_sg, 0, self.X_sg])

        phi_dn = q_dn / (self.C_mdn * self.A_m)
        phi_de = 0.
        phi_dg = q_dg / (self.C_mdg * self.A_m)
        phi_se = ( - self.dx * self.A_i * I_n_diff + self.A_i * sigma_n * phi_dn - self.A_i * sigma_n * q_sn / (self.C_msn * self.A_m) \
            - self.dx * self.A_i * I_g_diff + self.A_i * sigma_g * phi_dg - self.A_i * sigma_g * q_sg / (self.C_msg * self.A_m) - self.dx * self.A_e * I_e_diff ) \
            / ( self.A_e * sigma_e + self.A_i * sigma_n + self.A_i * sigma_g )
        phi_sn = q_sn / (self.C_msn * self.A_m) + phi_se
        phi_sg = q_sg / (self.C_msg * self.A_m) + phi_se
        phi_msn = q_sn / (self.C_msn * self.A_m)
        phi_msg = q_sg / (self.C_msg * self.A_m)
        phi_mdn = phi_dn - phi_de
        phi_mdg = phi_dg - phi_de

        if extra:
            return phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg, \
                I_n_diff, I_g_diff, I_e_diff, sigma_n, sigma_g, sigma_e, q_dn, q_dg, q_sn, q_sg
        else:
            return phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg
        
    def dkdt(self):
        
        phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg  = self.membrane_potentials()
        E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = self.reversal_potentials()
        j_Na_msn = self.j_Na_sn(phi_msn, E_Na_sn)
        j_K_msn = self.j_K_sn(phi_msn, E_K_sn)
        j_Cl_msn = self.j_Cl_sn(phi_msn, E_Cl_sn)
        
        j_Na_msg = self.j_Na_sg(phi_msg, E_Na_sg)
        j_K_msg = self.j_K_sg(phi_msg, E_K_sg)
        j_Cl_msg = self.j_Cl_sg(phi_msg, E_Cl_sg)
        
        j_Na_mdn = self.j_Na_dn(phi_mdn, E_Na_dn)
        j_K_mdn = self.j_K_dn(phi_mdn, E_K_dn)    
        j_Cl_mdn = self.j_Cl_dn(phi_mdn, E_Cl_dn)
        
        j_Na_mdg = self.j_Na_dg(phi_mdg, E_Na_dg)
        j_K_mdg = self.j_K_dg(phi_mdg, E_K_dg)
        j_Cl_mdg = self.j_Cl_dg(phi_mdg, E_Cl_dg)
        
        j_Ca_mdn = self.j_Ca_dn(phi_mdn, E_Ca_dn)
        
        j_Na_in = self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_sn, self.cNa_dn) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn, phi_sn, phi_dn) 
        j_K_in = self.j_k_diff(self.D_K, self.lamda_i, self.cK_sn, self.cK_dn) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn, phi_sn, phi_dn)
        j_Cl_in = self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn, phi_sn, phi_dn)
        j_Ca_in = self.j_k_diff(self.D_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn, phi_sn, phi_dn)

        j_Na_ig = self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_sg, self.cNa_dg) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, phi_sg, phi_dg) 
        j_K_ig = self.j_k_diff(self.D_K, self.lamda_i, self.cK_sg, self.cK_dg) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg, phi_sg, phi_dg)
        j_Cl_ig = self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg, phi_sg, phi_dg)

        j_Na_e = self.j_k_diff(self.D_Na, self.lamda_e, self.cNa_se, self.cNa_de) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, phi_se, phi_de)
        j_K_e = self.j_k_diff(self.D_K, self.lamda_e, self.cK_se, self.cK_de) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, phi_se, phi_de)
        j_Cl_e = self.j_k_diff(self.D_Cl, self.lamda_e, self.cCl_se, self.cCl_de) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, phi_se, phi_de)
        j_Ca_e = self.j_k_diff(self.D_Ca, self.lamda_e, self.cCa_se, self.cCa_de) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, phi_se, phi_de)
        
        dNadt_sn = -j_Na_msn*self.A_m - j_Na_in*self.A_i 
        dNadt_se = j_Na_msn*self.A_m + j_Na_msg*self.A_m - j_Na_e*self.A_e 
        dNadt_sg = -j_Na_msg*self.A_m - j_Na_ig*self.A_i
        dNadt_dn = -j_Na_mdn*self.A_m + j_Na_in*self.A_i 
        dNadt_de = j_Na_mdn*self.A_m + j_Na_mdg*self.A_m + j_Na_e*self.A_e 
        dNadt_dg = -j_Na_mdg*self.A_m + j_Na_ig*self.A_i

        dKdt_sn = -j_K_msn*self.A_m - j_K_in*self.A_i
        dKdt_se = j_K_msn*self.A_m + j_K_msg*self.A_m - j_K_e*self.A_e
        dKdt_sg = -j_K_msg*self.A_m - j_K_ig*self.A_i
        dKdt_dn = -j_K_mdn*self.A_m + j_K_in*self.A_i
        dKdt_de = j_K_mdn*self.A_m + j_K_mdg*self.A_m + j_K_e*self.A_e
        dKdt_dg = -j_K_mdg*self.A_m + j_K_ig*self.A_i

        dCldt_sn = -j_Cl_msn*self.A_m - j_Cl_in*self.A_i
        dCldt_se = j_Cl_msn*self.A_m + j_Cl_msg*self.A_m - j_Cl_e*self.A_e
        dCldt_sg = -j_Cl_msg*self.A_m - j_Cl_ig*self.A_i
        dCldt_dn = -j_Cl_mdn*self.A_m + j_Cl_in*self.A_i
        dCldt_de = j_Cl_mdn*self.A_m + j_Cl_mdg*self.A_m + j_Cl_e*self.A_e
        dCldt_dg = -j_Cl_mdg*self.A_m + j_Cl_ig*self.A_i

        dCadt_sn = - j_Ca_in*self.A_i - self.j_Ca_sn()*self.A_m
        dCadt_se = - j_Ca_e*self.A_e + self.j_Ca_sn()*self.A_m
        dCadt_dn = j_Ca_in*self.A_i - j_Ca_mdn*self.A_m 
        dCadt_de = j_Ca_e*self.A_e + j_Ca_mdn*self.A_m 

        return dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, \
            dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de

    def dmdt(self):
        phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg  = self.membrane_potentials()
        
        dndt = self.alpha_n(phi_msn)*(1.0-self.n) - self.beta_n(phi_msn)*self.n
        dhdt = self.alpha_h(phi_msn)*(1.0-self.h) - self.beta_h(phi_msn)*self.h 
        dsdt = self.alpha_s(phi_mdn)*(1.0-self.s) - self.beta_s(phi_mdn)*self.s
        dcdt = self.alpha_c(phi_mdn)*(1.0-self.c) - self.beta_c(phi_mdn)*self.c
        dqdt = self.alpha_q()*(1.0-self.q) - self.beta_q()*self.q
        dzdt = (self.z_inf(phi_mdn) - self.z)/1000.0
        
        return dndt, dhdt, dsdt, dcdt, dqdt, dzdt

    def dVdt(self):

        dVsidt = self.G_n * (self.psi_se - self.psi_sn)
        dVsgdt = self.G_g * (self.psi_se - self.psi_sg)
        dVdidt = self.G_n * (self.psi_de - self.psi_dn)
        dVdgdt = self.G_g * (self.psi_de - self.psi_dg)
        dVsedt = - (dVsidt + dVsgdt)
        dVdedt = - (dVdidt + dVdgdt)

        return dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt
    
    # Derivative functions
    def dalpha_m(self, phi_m, dphi_m):
        phi_1 = phi_m + 46.9
        alpha = 0.32 * np.exp(0.25*phi_1) *(-0.25*phi_1+np.exp(0.25*phi_1)-1)*dphi_m / (np.exp( 0.25 * phi_1) - 1.)**2
        return alpha

    def dbeta_m(self, phi_m, dphi_m):
        phi_2 = phi_m + 19.9
        beta = - 0.2 * 0.28 * (np.exp(0.2*phi_2)*phi_2 - 5 * np.exp(0.2*phi_2) + 5) * dphi_m / (np.exp(0.2 * phi_2) - 1)**2
        return beta

    def dalpha_h(self, phi_m, dphi_m):
        alpha = -0.128 * np.exp((-43 - phi_m) / 18) *dphi_m / 18
        return alpha

    def dbeta_h(self, phi_m, dphi_m):
        phi_3 = phi_m + 20
        beta = 4*0.2 * np.exp(-0.2*phi_3) * dphi_m / (1 + np.exp(-0.2*phi_3))**2
        return beta

    def dalpha_n(self, phi_m, dphi_m):
        phi_4 = phi_m + 24.9
        alpha = 0.016 * np.exp(0.2 * phi_4) * (-0.2*phi_4 + np.exp(0.2*phi_4)-1)*dphi_m / (np.exp(0.2 * phi_4) - 1)**2
        return alpha

    def dbeta_n(self, phi_m, dphi_m):
        phi_5 = phi_m + 40
        beta = -0.25/40 * np.exp(-phi_5 / 40) * dphi_m
        return beta

    def dalpha_s(self, phi_m, dphi_m):
        alpha = 1.6 * 0.072 * np.exp(-0.072 * (phi_m - 5)) * dphi_m / (1 + np.exp(-0.072 * (phi_m - 5)))**2
        return alpha

    def dbeta_s(self, phi_m, dphi_m):
        phi_6 = phi_m + 8.9
        beta = -0.02/5*(phi_6*np.exp(phi_6/5)-5*np.exp(phi_6/5)+5)*dphi_m/ (np.exp(0.2 * phi_6) - 1.)**2
        return beta

    def dalpha_c(self, phi_m, dphi_m):
        phi_8 = phi_m + 50
        phi_9 = phi_m + 53.5
        if phi_m <= -10:
            alpha = 0.0527 * np.exp(phi_8/11- phi_9/27) * (dphi_m/11 - dphi_m/27)
        else:
            alpha = 2 * np.exp(-phi_9 / 27) * (-dphi_m/27)
        return alpha

    def dbeta_c(self, phi_m, dphi_m):
        phi_9 = phi_m + 53.5
        if phi_m <= -10:
            beta = 2 * np.exp(-phi_9 / 27) * (-dphi_m/27) - self.dalpha_c(phi_m, dphi_m)
        else:
            beta = 0.
        return beta

    def dchi(self, dfree_cCa_dn_dk):
        if (self.free_cCa_dn-99.8e-3)/2.5e-1 <= 1.0:
            chi = dfree_cCa_dn_dk / 2.5e-1
        else:
            chi = 0.
        return chi

    def dchi_dCa_dn(self):  
        return 0.04*np.heaviside(-0.04*self.Ca_dn/self.V_dn + 1.3992, 0.5)/self.V_dn 
    
    def dchi_dV_dn(self):  
        return -0.04*self.Ca_dn*np.heaviside(-0.04*self.Ca_dn/self.V_dn + 1.3992, 0.5)/self.V_dn**2
    
    def dalpha_q_dCa_dn(self):        
        return 0.0002*np.heaviside(-0.2*self.Ca_dn/self.V_dn + 11.996, 0.5)/self.V_dn

    def dalpha_q_dV_dn(self):        
        return -0.0002*self.Ca_dn*np.heaviside(-0.2*self.Ca_dn/self.V_dn + 11.996, 0.5)/self.V_dn**2

    def dm_inf(self, phi_m, dphi_m):
        return (self.beta_m(phi_m) * self.dalpha_m(phi_m, dphi_m)-self.alpha_m(phi_m) * self.dbeta_m(phi_m, dphi_m)) / (self.alpha_m(phi_m) + self.beta_m(phi_m))**2

    def dz_inf(self, phi_m, dphi_m):
        phi_7 = phi_m + 30
        return - np.exp(phi_7) * dphi_m /(1 + np.exp(phi_7))**2

    def dj_pump_n_dn(self, cNa_n, cK_e, dcNa_n):
        j = (self.rho_n *  np.exp((25000. - cNa_n)/3000.) * dcNa_n / (3000 * (1.0 + np.exp((25000. - cNa_n)/3000.))**2)) * (1.0 / (1.0 + np.exp((3500 - cK_e)/1000)))
        return j

    def dj_pump_n_de(self, cNa_n, cK_e, dcK_e):
        j = (self.rho_n / (1.0 + np.exp((25000. - cNa_n)/3000.))) * (np.exp((3500 - cK_e)/1000)) * dcK_e / (1000 * (1.0 + np.exp((3500 - cK_e)/1000))**2)
        return j

    def dj_pump_g_dg(self, cNa_g, cK_e, dcNa_g):
        j = self.rho_g * (cK_e / (cK_e + self.cK_treshold)) * ( 1.5 * cNa_g**0.5 * dcNa_g / (cNa_g**1.5 + self.cNa_treshold**1.5) - 1.5 * cNa_g**2 * dcNa_g / (cNa_g**1.5 + self.cNa_treshold**1.5)**2)
        return j

    def dj_pump_g_de(self, cNa_g, cK_e, dcK_e):
        j = self.rho_g * (cNa_g**1.5 / (cNa_g**1.5 + self.cNa_treshold**1.5)) * (self.cK_treshold * dcK_e / (cK_e + self.cK_treshold)**2)
        return j

    def dj_kcc2_dn(self, c_n, dc_n):
        j = self.U_kcc2 * dc_n / c_n
        return j

    def dj_kcc2_de(self, c_e, dc_e):
        j = - self.U_kcc2 * dc_e/c_e
        return j

    def dj_kcc2_dVn(self, cK_n, cCl_n, dcK_n, dcCl_n):
        j = self.U_kcc2 * (cK_n * dcCl_n + dcK_n * cCl_n) / (cK_n * cCl_n)
        return j

    def dj_kcc2_dVe(self, cK_e, cCl_e, dcK_e, dcCl_e):
        j = - self.U_kcc2 * (cK_e * dcCl_e + dcK_e * cCl_e) / (cK_e * cCl_e)
        return j

    def dj_nkcc1_dn(self, c_n, dc_n, cK_e):
        j = self.U_nkcc1 * (1 / (1 + np.exp((16e3 - cK_e)/1e3))) * dc_n / c_n
        return j

    def dj_nkcc1_de(self, c_e, dc_e, cK_e):
        j = - self.U_nkcc1 * (1 / (1 + np.exp((16e3 - cK_e)/1e3))) * dc_e/c_e
        return j

    def dj_nkcc1_dKe(self, cNa_n, cNa_e, cK_n, cK_e, cCl_n, cCl_e, dcK_e):
        j = - self.U_nkcc1 * (1 / (1 + np.exp((16e3 - cK_e)/1e3))) * dcK_e/cK_e + self.U_nkcc1 *  (np.log(cK_n*cCl_n/(cK_e*cCl_e)) + np.log(cNa_n*cCl_n/(cNa_e*cCl_e))) * (np.exp((16e3 - cK_e)/1e3) * (dcK_e/1e3) / (1 + np.exp((16e3 - cK_e)/1e3))**2) 
        return j

    def dj_nkcc1_dVn(self, cNa_n, cK_n, cCl_n, dcNa_n, dcK_n, dcCl_n, cK_e):
        j = self.U_nkcc1 * (1 / (1 + np.exp((16e3 - cK_e)/1e3))) * ((cK_n * dcCl_n + dcK_n * cCl_n) / (cK_n * cCl_n) + (cNa_n * dcCl_n + dcNa_n * cCl_n) / (cNa_n * cCl_n))
        return j

    def dj_nkcc1_dVe(self, cNa_n, cNa_e, cK_n, cK_e, cCl_n, cCl_e, dcNa_e, dcK_e, dcCl_e):
        j = - self.dj_nkcc1_dVn(cNa_e, cK_e, cCl_e, dcNa_e, dcK_e, dcCl_e, cK_e) \
            + self.U_nkcc1 * (np.log(cK_n*cCl_n/(cK_e*cCl_e)) + np.log(cNa_n*cCl_n/(cNa_e*cCl_e))) * (np.exp((16e3 - cK_e)/1e3) * (dcK_e/1e3) / (1 + np.exp((16e3 - cK_e)/1e3))**2) 
        return j

    def dj_k_diff(self, D_k, tortuosity, dck):
        j = - D_k * (dck) / (tortuosity**2 * self.dx)
        return j

    def dj_k_drift_dck(self, D_k, Z_k, tortuosity, phi_s, phi_d, dck):
        j1 = - D_k * self.F * Z_k / (2 * tortuosity**2 * self.R * self.T * self.dx) * (phi_d - phi_s) * (dck)
        return j1 
    def dj_k_drift_dphi_s(self, D_k, Z_k, tortuosity,ck_s, ck_d, dphi):
        j2 =  D_k * self.F * Z_k / (2 * tortuosity**2 * self.R * self.T * self.dx) * (ck_d + ck_s) * (dphi)
        return j2
    def dj_k_drift_dphi_d(self, D_k, Z_k, tortuosity,ck_s, ck_d, dphi_d):
        j3 = - D_k * self.F * Z_k / (2 * tortuosity**2 * self.R * self.T * self.dx) * (ck_d + ck_s) * (dphi_d)
        return j3
    
    def dconductivity_k(self, D_k, Z_k, tortuosity, dck): 
            sigma = self.F**2 * D_k * Z_k**2 * (dck) / (2 * self.R * self.T * tortuosity**2)
            return sigma

    def dnernst_potential(self, Z, ck, dck):        # Sign changed when used (- for internal compartments)
            E = self.R*self.T / (Z*self.F*ck) * dck
            return E

    def dpsi_dck(self, dck):
            return -self.R*self.T*dck
    
    def df_dphi_m(self, phi_m, E_K, dphi_m, flag):
        dphi = (phi_m - E_K)
        phi_m_mil = phi_m
        if flag == 's':
            bE_K_mil = self.bE_K_sg
        elif flag == 'd':
            bE_K_mil = self.bE_K_dg
        fact1 = (1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))
        fact2 = (1 + np.exp(-(118.6+bE_K_mil)/44.1))/(1+np.exp(-(118.6+phi_m_mil)/44.1))
        dfact1 = - 1/42.5 * (1 + np.exp(18.4/42.4))*np.exp((dphi + 18.5)/42.5) * dphi_m /(1 + np.exp((dphi + 18.5)/42.5))**2
        dfact2 = 1/44.1 * (1 + np.exp(-(118.6+bE_K_mil)/44.1))*np.exp(-(118.6+phi_m_mil)/44.1) * dphi_m /(1+np.exp(-(118.6+phi_m_mil)/44.1))**2
        f = dfact1 * fact2 + fact1 * dfact2
        return f

    def df_dE_K(self, phi_m, E_K, dE_K, flag):
        dphi = (phi_m - E_K)
        ddphi_dE_K = -dE_K
        phi_m_mil = phi_m
        if flag == 's':
            bE_K_mil = self.bE_K_sg
        elif flag == 'd':
            bE_K_mil = self.bE_K_dg
        fact2 = (1 + np.exp(-(118.6+bE_K_mil)/44.1))/(1+np.exp(-(118.6+phi_m_mil)/44.1))
        dfact1 = - 1/42.5 * (1 + np.exp(18.4/42.4))*np.exp((dphi + 18.5)/42.5) * ddphi_dE_K /(1 + np.exp((dphi + 18.5)/42.5))**2
        f = dfact1 * fact2
        return f

    def df_sqrt(self, cK, dcK, cbK_se):
        f = dcK / (2 * np.sqrt(cK * cbK_se))
        return f

    def f(self, phi_m, E_K, flag):
        dphi = (phi_m - E_K)
        phi_m_mil = phi_m
        if flag == 's':
            bE_K_mil = self.bE_K_sg
        elif flag == 'd':
            bE_K_mil = self.bE_K_dg
        fact1 = (1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))
        fact2 = (1 + np.exp(-(118.6+bE_K_mil)/44.1))/(1+np.exp(-(118.6+phi_m_mil)/44.1))
        return fact1*fact2

    def edNEG_jacobian(self, dense=False):      
        E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = self.reversal_potentials()

        phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg, \
                I_n_diff, I_g_diff, I_e_diff, sigma_n, sigma_g, sigma_e, q_dn, q_dg, q_sn, q_sg = self.membrane_potentials(extra=True)

        # Concentrations derivatives
        dcNa_sn_dNasn   = dcK_sn_dKsn     = dcCl_sn_dClsn   = dcCa_sn_dCasn   = 1/self.V_sn
        dfree_cCa_sn_dCasn = 0.01*dcCa_sn_dCasn
        dcNa_se_dNase   = dcK_se_dKse     = dcCl_se_dClse   = dcCa_se_dCase   = 1/self.V_se
        dcNa_sg_dNasg   = dcK_sg_dKsg     = dcCl_sg_dClsg   = 1/self.V_sg
        dcNa_dn_dNadn   = dcK_dn_dKdn     = dcCl_dn_dCldn   = dcCa_dn_dCadn   = 1/self.V_dn
        dfree_cCa_dn_dCadn = 0.01*dcCa_dn_dCadn
        dcNa_de_dNade   = dcK_de_dKde     = dcCl_de_dClde   = dcCa_de_dCade   = 1/self.V_de
        dcNa_dg_dNadg   = dcK_dg_dKdg     = dcCl_dg_dCldg   = 1/self.V_dg

        dcNa_sn_dVsn    = -self.Na_sn/self.V_sn**2
        dcK_sn_dVsn     = -self.K_sn/self.V_sn**2
        dcCl_sn_dVsn    = -self.Cl_sn/self.V_sn**2
        dcCa_sn_dVsn    = -self.Ca_sn/self.V_sn**2
        dfree_cCa_sn_dVsn = 0.01 * dcCa_sn_dVsn

        dcNa_se_dVse    = -self.Na_se/self.V_se**2
        dcK_se_dVse     = -self.K_se/self.V_se**2
        dcCl_se_dVse    = -self.Cl_se/self.V_se**2
        dcCa_se_dVse    = -self.Ca_se/self.V_se**2

        dcNa_sg_dVsg    = -self.Na_sg/self.V_sg**2
        dcK_sg_dVsg     = -self.K_sg/self.V_sg**2
        dcCl_sg_dVsg    = -self.Cl_sg/self.V_sg**2

        dcNa_dn_dVdn    = -self.Na_dn/self.V_dn**2
        dcK_dn_dVdn     = -self.K_dn/self.V_dn**2
        dcCl_dn_dVdn    = -self.Cl_dn/self.V_dn**2
        dcCa_dn_dVdn    = -self.Ca_dn/self.V_dn**2
        dfree_cCa_dn_dVdn = 0.01 * dcCa_dn_dVdn

        dcNa_de_dVde    = -self.Na_de/self.V_de**2
        dcK_de_dVde     = -self.K_de/self.V_de**2
        dcCl_de_dVde    = -self.Cl_de/self.V_de**2
        dcCa_de_dVde    = -self.Ca_de/self.V_de**2

        dcNa_dg_dVdg    = -self.Na_dg/self.V_dg**2
        dcK_dg_dVdg     = -self.K_dg/self.V_dg**2
        dcCl_dg_dVdg    = -self.Cl_dg/self.V_dg**2
        
        # Reversal potentials derivatives
        dE_Na_sn_dNasn = - self.dnernst_potential(self.Z_Na, self.cNa_sn, dcNa_sn_dNasn)
        dE_Na_sn_dVsn = - self.dnernst_potential(self.Z_Na, self.cNa_sn, dcNa_sn_dVsn)
        dE_Na_sg_dNasg = - self.dnernst_potential(self.Z_Na, self.cNa_sg, dcNa_sg_dNasg)
        dE_Na_sg_dVsg = - self.dnernst_potential(self.Z_Na, self.cNa_sg, dcNa_sg_dVsg)
        dE_Na_dn_dNadn = - self.dnernst_potential(self.Z_Na, self.cNa_dn, dcNa_dn_dNadn)
        dE_Na_dn_dVdn = - self.dnernst_potential(self.Z_Na, self.cNa_dn, dcNa_dn_dVdn)
        dE_Na_dg_dNadg = - self.dnernst_potential(self.Z_Na, self.cNa_dg, dcNa_dg_dNadg)
        dE_Na_dg_dVdg = - self.dnernst_potential(self.Z_Na, self.cNa_dg, dcNa_dg_dVdg)
        dE_K_sn_dKsn = - self.dnernst_potential(self.Z_K, self.cK_sn, dcK_sn_dKsn)
        dE_K_sn_dVsn = - self.dnernst_potential(self.Z_K, self.cK_sn, dcK_sn_dVsn)
        dE_K_sg_dKsg = - self.dnernst_potential(self.Z_K, self.cK_sg, dcK_sg_dKsg)
        dE_K_sg_dVsg = - self.dnernst_potential(self.Z_K, self.cK_sg, dcK_sg_dVsg)
        dE_K_dn_dKdn = - self.dnernst_potential(self.Z_K, self.cK_dn, dcK_dn_dKdn)
        dE_K_dn_dVdn = - self.dnernst_potential(self.Z_K, self.cK_dn, dcK_dn_dVdn)
        dE_K_dg_dKdg = - self.dnernst_potential(self.Z_K, self.cK_dg, dcK_dg_dKdg)
        dE_K_dg_dVdg = - self.dnernst_potential(self.Z_K, self.cK_dg, dcK_dg_dVdg)
        dE_Cl_sn_dClsn = - self.dnernst_potential(self.Z_Cl, self.cCl_sn, dcCl_sn_dClsn)
        dE_Cl_sn_dVsn = - self.dnernst_potential(self.Z_Cl, self.cCl_sn, dcCl_sn_dVsn)
        dE_Cl_sg_dClsg = - self.dnernst_potential(self.Z_Cl, self.cCl_sg, dcCl_sg_dClsg)
        dE_Cl_sg_dVsg = - self.dnernst_potential(self.Z_Cl, self.cCl_sg, dcCl_sg_dVsg)
        dE_Cl_dn_dCldn = - self.dnernst_potential(self.Z_Cl, self.cCl_dn, dcCl_dn_dCldn)
        dE_Cl_dn_dVdn = - self.dnernst_potential(self.Z_Cl, self.cCl_dn, dcCl_dn_dVdn)
        dE_Cl_dg_dCldg = - self.dnernst_potential(self.Z_Cl, self.cCl_dg, dcCl_dg_dCldg)
        dE_Cl_dg_dVdg = - self.dnernst_potential(self.Z_Cl, self.cCl_dg, dcCl_dg_dVdg)
        dE_Ca_sn_dCasn = - self.dnernst_potential(self.Z_Ca, self.free_cCa_sn, dfree_cCa_sn_dCasn)
        dE_Ca_sn_dVsn = - self.dnernst_potential(self.Z_Ca, self.free_cCa_sn, dfree_cCa_sn_dVsn)
        dE_Ca_dn_dCadn = - self.dnernst_potential(self.Z_Ca, self.free_cCa_dn, dfree_cCa_dn_dCadn)
        dE_Ca_dn_dVdn = - self.dnernst_potential(self.Z_Ca, self.free_cCa_dn, dfree_cCa_dn_dVdn)

        dE_Na_sn_dNase = dE_Na_sg_dNase = self.dnernst_potential(self.Z_Na, self.cNa_se, dcNa_se_dNase)
        dE_Na_sn_dVse = dE_Na_sg_dVse = self.dnernst_potential(self.Z_Na, self.cNa_se, dcNa_se_dVse)
        dE_Na_dn_dNade = dE_Na_dg_dNade = self.dnernst_potential(self.Z_Na, self.cNa_de, dcNa_de_dNade) 
        dE_Na_dn_dVde = dE_Na_dg_dVde = self.dnernst_potential(self.Z_Na, self.cNa_de, dcNa_de_dVde) 
        dE_K_sn_dKse = dE_K_sg_dKse = self.dnernst_potential(self.Z_K, self.cK_se, dcK_se_dKse)
        dE_K_sn_dVse = dE_K_sg_dVse = self.dnernst_potential(self.Z_K, self.cK_se, dcK_se_dVse)
        dE_K_dn_dKde = dE_K_dg_dKde = self.dnernst_potential(self.Z_K, self.cK_de, dcK_de_dKde)
        dE_K_dn_dVde = dE_K_dg_dVde = self.dnernst_potential(self.Z_K, self.cK_de, dcK_de_dVde)
        dE_Cl_sn_dClse = dE_Cl_sg_dClse = self.dnernst_potential(self.Z_Cl, self.cCl_se, dcCl_se_dClse)
        dE_Cl_sn_dVse = dE_Cl_sg_dVse = self.dnernst_potential(self.Z_Cl, self.cCl_se, dcCl_se_dVse)
        dE_Cl_dn_dClde = dE_Cl_dg_dClde = self.dnernst_potential(self.Z_Cl, self.cCl_de, dcCl_de_dClde)
        dE_Cl_dn_dVde = dE_Cl_dg_dVde = self.dnernst_potential(self.Z_Cl, self.cCl_de, dcCl_de_dVde)
        dE_Ca_sn_dCase = self.dnernst_potential(self.Z_Ca, self.cCa_se, dcCa_se_dCase)
        dE_Ca_sn_dVse = self.dnernst_potential(self.Z_Ca, self.cCa_se, dcCa_se_dVse)
        dE_Ca_dn_dCade = self.dnernst_potential(self.Z_Ca, self.cCa_de, dcCa_de_dCade)
        dE_Ca_dn_dVde = self.dnernst_potential(self.Z_Ca, self.cCa_de, dcCa_de_dVde)


        # Currents derivatives
        
        # I_n_diff
        dI_n_diff_dNasn = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_i, -dcNa_sn_dNasn))
        dI_n_diff_dKsn = self.F * (self.Z_K*self.dj_k_diff(self.D_K, self.lamda_i, -dcK_sn_dKsn))
        dI_n_diff_dClsn = self.F * (self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_i, -dcCl_sn_dClsn))
        dI_n_diff_dCasn = self.F * (self.Z_Ca*self.dj_k_diff(self.D_Ca, self.lamda_i, -dfree_cCa_sn_dCasn))

        dI_n_diff_dNadn = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_i, dcNa_dn_dNadn))
        dI_n_diff_dKdn = self.F * (self.Z_K*self.dj_k_diff(self.D_K, self.lamda_i, dcK_dn_dKdn))
        dI_n_diff_dCldn = self.F * (self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_i, dcCl_dn_dCldn))
        dI_n_diff_dCadn = self.F * (self.Z_Ca*self.dj_k_diff(self.D_Ca, self.lamda_i, dfree_cCa_dn_dCadn))

        dI_n_diff_dVsn = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_i, -dcNa_sn_dVsn) \
            + self.Z_K*self.dj_k_diff(self.D_K, self.lamda_i, -dcK_sn_dVsn) \
            + self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_i, -dcCl_sn_dVsn) \
            + self.Z_Ca*self.dj_k_diff(self.D_Ca, self.lamda_i, -dfree_cCa_sn_dVsn))

        dI_n_diff_dVdn = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_i, dcNa_dn_dVdn) \
            + self.Z_K*self.dj_k_diff(self.D_K, self.lamda_i, dcK_dn_dVdn) \
            + self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_i, dcCl_dn_dVdn) \
            + self.Z_Ca*self.dj_k_diff(self.D_Ca, self.lamda_i, dfree_cCa_dn_dVdn))    

        # I_g_diff
        dI_g_diff_dNasg = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_i, -dcNa_sg_dNasg))
        dI_g_diff_dKsg = self.F * (self.Z_K*self.dj_k_diff(self.D_K, self.lamda_i, -dcK_sg_dKsg))
        dI_g_diff_dClsg = self.F * (self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_i, -dcCl_sg_dClsg))

        dI_g_diff_dNadg = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_i, dcNa_dg_dNadg))
        dI_g_diff_dKdg = self.F * (self.Z_K*self.dj_k_diff(self.D_K, self.lamda_i, dcK_dg_dKdg))
        dI_g_diff_dCldg = self.F * (self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_i, dcCl_dg_dCldg))

        dI_g_diff_dVsg = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_i, -dcNa_sg_dVsg) \
            + self.Z_K*self.dj_k_diff(self.D_K, self.lamda_i, -dcK_sg_dVsg) \
            + self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_i, -dcCl_sg_dVsg))

        dI_g_diff_dVdg = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_i, dcNa_dg_dVdg) \
            + self.Z_K*self.dj_k_diff(self.D_K, self.lamda_i, dcK_dg_dVdg) \
            + self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_i, dcCl_dg_dVdg)) 

        # I_e_diff
        dI_e_diff_dNase = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_e, -dcNa_se_dNase))
        dI_e_diff_dKse = self.F * (self.Z_K*self.dj_k_diff(self.D_K, self.lamda_e, -dcK_se_dKse))
        dI_e_diff_dClse = self.F * (self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_e, -dcCl_se_dClse))
        dI_e_diff_dCase = self.F * (self.Z_Ca*self.dj_k_diff(self.D_Ca, self.lamda_e, -dcCa_se_dCase))

        dI_e_diff_dNade = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_e, dcNa_de_dNade))
        dI_e_diff_dKde = self.F * (self.Z_K*self.dj_k_diff(self.D_K, self.lamda_e, dcK_de_dKde))
        dI_e_diff_dClde = self.F * (self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_e, dcCl_de_dClde))
        dI_e_diff_dCade = self.F * (self.Z_Ca*self.dj_k_diff(self.D_Ca, self.lamda_e, dcCa_de_dCade))

        dI_e_diff_dVse = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_e, -dcNa_se_dVse) \
            + self.Z_K*self.dj_k_diff(self.D_K, self.lamda_e, -dcK_se_dVse) \
            + self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_e, -dcCl_se_dVse) \
            + self.Z_Ca*self.dj_k_diff(self.D_Ca, self.lamda_e, -dcCa_se_dVse))

        dI_e_diff_dVde = self.F * (self.Z_Na*self.dj_k_diff(self.D_Na, self.lamda_e, dcNa_de_dVde) \
            + self.Z_K*self.dj_k_diff(self.D_K, self.lamda_e, dcK_de_dVde) \
            + self.Z_Cl*self.dj_k_diff(self.D_Cl, self.lamda_e, dcCl_de_dVde) \
            + self.Z_Ca*self.dj_k_diff(self.D_Ca, self.lamda_e, dcCa_de_dVde))

        # Membrane potentials derivatives

        # sigma_n
        dsigma_n_dNasn = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_i, dcNa_sn_dNasn)
        dsigma_n_dKsn = self.dconductivity_k(self.D_K, self.Z_K, self.lamda_i, dcK_sn_dKsn)
        dsigma_n_dClsn = self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, dcCl_sn_dClsn)
        dsigma_n_dCasn = self.dconductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, dfree_cCa_sn_dCasn)

        dsigma_n_dNadn = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_i, dcNa_dn_dNadn)
        dsigma_n_dKdn = self.dconductivity_k(self.D_K, self.Z_K, self.lamda_i, dcK_dn_dKdn)
        dsigma_n_dCldn = self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, dcCl_dn_dCldn)
        dsigma_n_dCadn = self.dconductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, dfree_cCa_dn_dCadn)

        dsigma_n_dVsn = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_i, dcNa_sn_dVsn) \
            + self.dconductivity_k(self.D_K, self.Z_K, self.lamda_i, dcK_sn_dVsn) \
            + self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, dcCl_sn_dVsn) \
            + self.dconductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, dfree_cCa_sn_dVsn)

        dsigma_n_dVdn = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_i, dcNa_dn_dVdn) \
            + self.dconductivity_k(self.D_K, self.Z_K, self.lamda_i, dcK_dn_dVdn) \
            + self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, dcCl_dn_dVdn) \
            + self.dconductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, dfree_cCa_dn_dVdn)

        # sigma_g
        dsigma_g_dNasg = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_i, dcNa_sg_dNasg)
        dsigma_g_dKsg = self.dconductivity_k(self.D_K, self.Z_K, self.lamda_i, dcK_sg_dKsg)
        dsigma_g_dClsg = self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, dcCl_sg_dClsg)

        dsigma_g_dNadg = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_i, dcNa_dg_dNadg)
        dsigma_g_dKdg = self.dconductivity_k(self.D_K, self.Z_K, self.lamda_i, dcK_dg_dKdg)
        dsigma_g_dCldg = self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, dcCl_dg_dCldg)

        dsigma_g_dVsg = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_i, dcNa_sg_dVsg) \
            + self.dconductivity_k(self.D_K, self.Z_K, self.lamda_i, dcK_sg_dVsg) \
            + self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, dcCl_sg_dVsg)

        dsigma_g_dVdg = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_i, dcNa_dg_dVdg) \
            + self.dconductivity_k(self.D_K, self.Z_K, self.lamda_i, dcK_dg_dVdg) \
            + self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, dcCl_dg_dVdg)

        # sigma_e
        dsigma_e_dNase = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_e, dcNa_se_dNase)
        dsigma_e_dKse = self.dconductivity_k(self.D_K, self.Z_K, self.lamda_e, dcK_se_dKse)
        dsigma_e_dClse = self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, dcCl_se_dClse)
        dsigma_e_dCase = self.dconductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, dcCa_se_dCase)

        dsigma_e_dNade = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_e, dcNa_de_dNade)
        dsigma_e_dKde = self.dconductivity_k(self.D_K, self.Z_K, self.lamda_e, dcK_de_dKde)
        dsigma_e_dClde = self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, dcCl_de_dClde)
        dsigma_e_dCade = self.dconductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, dcCa_de_dCade)

        dsigma_e_dVse = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_e, dcNa_se_dVse) \
            + self.dconductivity_k(self.D_K, self.Z_K, self.lamda_e, dcK_se_dVse) \
            + self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, dcCl_se_dVse) \
            + self.dconductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, dcCa_se_dVse)

        dsigma_e_dVde = self.dconductivity_k(self.D_Na, self.Z_Na, self.lamda_e, dcNa_de_dVde) \
            + self.dconductivity_k(self.D_K, self.Z_K, self.lamda_e, dcK_de_dVde) \
            + self.dconductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, dcCl_de_dVde) \
            + self.dconductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, dcCa_de_dVde)

        # q_dn, q_dg, q_sn, q_sg
        dq_dn_dNadn = dq_dg_dNadg = dq_sn_dNasn = dq_sg_dNasg = self.Z_Na * self.F
        dq_dn_dKdn = dq_dg_dKdg = dq_sn_dKsn = dq_sg_dKsg = self.Z_K * self.F
        dq_dn_dCldn = dq_dg_dCldg = dq_sn_dClsn = dq_sg_dClsg = self.Z_Cl * self.F
        dq_dn_dCadn = dq_sn_dCasn = self.Z_Ca * self.F

        # phi_dn
        dphi_dn_dNadn = dq_dn_dNadn / (self.C_mdn * self.A_m)
        dphi_dn_dKdn = dq_dn_dKdn / (self.C_mdn * self.A_m)
        dphi_dn_dCldn = dq_dn_dCldn / (self.C_mdn * self.A_m)
        dphi_dn_dCadn = dq_dn_dCadn / (self.C_mdn * self.A_m)

        # phi_dg
        dphi_dg_dNadg = dq_dg_dNadg / (self.C_mdg * self.A_m)
        dphi_dg_dKdg = dq_dg_dKdg / (self.C_mdg * self.A_m)
        dphi_dg_dCldg = dq_dg_dCldg / (self.C_mdg * self.A_m)

        # phi_se
        phi_se_num = ( - self.dx * self.A_i * I_n_diff + self.A_i * sigma_n * phi_dn - self.A_i * sigma_n * q_sn / (self.C_msn * self.A_m) \
            - self.dx * self.A_i * I_g_diff + self.A_i * sigma_g * phi_dg - self.A_i * sigma_g * q_sg / (self.C_msg * self.A_m) - self.dx * self.A_e * I_e_diff )
        phi_se_den = self.A_e * sigma_e + self.A_i * sigma_n + self.A_i * sigma_g 

        dphi_se_dNasn = ((- self.dx * self.A_i * dI_n_diff_dNasn + self.A_i * dsigma_n_dNasn * phi_dn - self.A_i / (self.C_msn * self.A_m) * (dsigma_n_dNasn * q_sn + sigma_n * dq_sn_dNasn)) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dNasn) / phi_se_den**2
        dphi_se_dKsn = ((- self.dx * self.A_i * dI_n_diff_dKsn + self.A_i * dsigma_n_dKsn * phi_dn - self.A_i / (self.C_msn * self.A_m) * (dsigma_n_dKsn * q_sn + sigma_n * dq_sn_dKsn)) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dKsn) / phi_se_den**2
        dphi_se_dClsn = ((- self.dx * self.A_i * dI_n_diff_dClsn + self.A_i * dsigma_n_dClsn * phi_dn - self.A_i / (self.C_msn * self.A_m) * (dsigma_n_dClsn * q_sn + sigma_n * dq_sn_dClsn)) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dClsn) / phi_se_den**2
        dphi_se_dCasn = ((- self.dx * self.A_i * dI_n_diff_dCasn + self.A_i * dsigma_n_dCasn * phi_dn - self.A_i / (self.C_msn * self.A_m) * (dsigma_n_dCasn * q_sn + sigma_n * dq_sn_dCasn)) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dCasn) / phi_se_den**2
        dphi_se_dVsn = ((- self.dx * self.A_i * dI_n_diff_dVsn + self.A_i * dsigma_n_dVsn * phi_dn - self.A_i / (self.C_msn * self.A_m) * dsigma_n_dVsn * q_sn) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dVsn) / phi_se_den**2

        dphi_se_dNadn = ((- self.dx * self.A_i * dI_n_diff_dNadn + self.A_i * (dsigma_n_dNadn * phi_dn + sigma_n * dphi_dn_dNadn) - self.A_i / (self.C_msn * self.A_m) * q_sn  * dsigma_n_dNadn) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dNadn) / phi_se_den**2
        dphi_se_dKdn = ((- self.dx * self.A_i * dI_n_diff_dKdn + self.A_i * (dsigma_n_dKdn * phi_dn + sigma_n * dphi_dn_dKdn) - self.A_i / (self.C_msn * self.A_m) * q_sn  * dsigma_n_dKdn) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dKdn) / phi_se_den**2
        dphi_se_dCldn = ((- self.dx * self.A_i * dI_n_diff_dCldn + self.A_i * (dsigma_n_dCldn * phi_dn + sigma_n * dphi_dn_dCldn) - self.A_i / (self.C_msn * self.A_m) * q_sn  * dsigma_n_dCldn) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dCldn) / phi_se_den**2
        dphi_se_dCadn = ((- self.dx * self.A_i * dI_n_diff_dCadn + self.A_i * (dsigma_n_dCadn * phi_dn + sigma_n * dphi_dn_dCadn) - self.A_i / (self.C_msn * self.A_m) * q_sn  * dsigma_n_dCadn) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dCadn) / phi_se_den**2
        dphi_se_dVdn = ((- self.dx * self.A_i * dI_n_diff_dVdn + self.A_i * dsigma_n_dVdn * phi_dn - self.A_i / (self.C_msn * self.A_m) * q_sn  * dsigma_n_dVdn) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_n_dVdn) / phi_se_den**2

        dphi_se_dNasg = ((- self.dx * self.A_i * dI_g_diff_dNasg + self.A_i * dsigma_g_dNasg * phi_dg - self.A_i / (self.C_msg * self.A_m) * (dsigma_g_dNasg * q_sg + sigma_g * dq_sg_dNasg)) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_g_dNasg) / phi_se_den**2
        dphi_se_dKsg = ((- self.dx * self.A_i * dI_g_diff_dKsg + self.A_i * dsigma_g_dKsg * phi_dg - self.A_i / (self.C_msg * self.A_m) * (dsigma_g_dKsg * q_sg + sigma_g * dq_sg_dKsg)) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_g_dKsg) / phi_se_den**2
        dphi_se_dClsg = ((- self.dx * self.A_i * dI_g_diff_dClsg + self.A_i * dsigma_g_dClsg * phi_dg - self.A_i / (self.C_msg * self.A_m) * (dsigma_g_dClsg * q_sg + sigma_g * dq_sg_dClsg)) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_g_dClsg) / phi_se_den**2
        dphi_se_dVsg = ((- self.dx * self.A_i * dI_g_diff_dVsg + self.A_i * dsigma_g_dVsg * phi_dg - self.A_i / (self.C_msg * self.A_m) * dsigma_g_dVsg * q_sg) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_g_dVsg) / phi_se_den**2

        dphi_se_dNadg = ((- self.dx * self.A_i * dI_g_diff_dNadg + self.A_i * (dsigma_g_dNadg * phi_dg + sigma_g * dphi_dg_dNadg) - self.A_i / (self.C_msg * self.A_m) * q_sg  * dsigma_g_dNadg) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_g_dNadg) / phi_se_den**2
        dphi_se_dKdg = ((- self.dx * self.A_i * dI_g_diff_dKdg + self.A_i * (dsigma_g_dKdg * phi_dg + sigma_g * dphi_dg_dKdg) - self.A_i / (self.C_msg * self.A_m) * q_sg  * dsigma_g_dKdg) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_g_dKdg) / phi_se_den**2
        dphi_se_dCldg = ((- self.dx * self.A_i * dI_g_diff_dCldg + self.A_i * (dsigma_g_dCldg * phi_dg + sigma_g * dphi_dg_dCldg) - self.A_i / (self.C_msg * self.A_m) * q_sg  * dsigma_g_dCldg) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_g_dCldg) / phi_se_den**2
        dphi_se_dVdg = ((- self.dx * self.A_i * dI_g_diff_dVdg + self.A_i * dsigma_g_dVdg * phi_dg - self.A_i / (self.C_msg * self.A_m) * q_sg  * dsigma_g_dVdg) * phi_se_den \
                        - phi_se_num * self.A_i * dsigma_g_dVdg) / phi_se_den**2

        dphi_se_dNase = ((- self.dx * self.A_e * dI_e_diff_dNase) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dNase) / phi_se_den**2
        dphi_se_dKse = ((- self.dx * self.A_e * dI_e_diff_dKse) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dKse) / phi_se_den**2
        dphi_se_dClse = ((- self.dx * self.A_e * dI_e_diff_dClse) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dClse) / phi_se_den**2
        dphi_se_dCase = ((- self.dx * self.A_e * dI_e_diff_dCase) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dCase) / phi_se_den**2
        dphi_se_dVse = ((- self.dx * self.A_e * dI_e_diff_dVse) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dVse) / phi_se_den**2

        dphi_se_dNade = ((- self.dx * self.A_e * dI_e_diff_dNade) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dNade) / phi_se_den**2
        dphi_se_dKde = ((- self.dx * self.A_e * dI_e_diff_dKde) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dKde) / phi_se_den**2
        dphi_se_dClde = ((- self.dx * self.A_e * dI_e_diff_dClde) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dClde) / phi_se_den**2
        dphi_se_dCade = ((- self.dx * self.A_e * dI_e_diff_dCade) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dCade) / phi_se_den**2
        dphi_se_dVde = ((- self.dx * self.A_e * dI_e_diff_dVde) * phi_se_den - phi_se_num * self.A_e * dsigma_e_dVde) / phi_se_den**2 

        # phi_sn
        dphi_sn_dNasn = dq_sn_dNasn / (self.C_msn * self.A_m) + dphi_se_dNasn
        dphi_sn_dKsn = dq_sn_dKsn / (self.C_msn * self.A_m) + dphi_se_dKsn 
        dphi_sn_dClsn = dq_sn_dClsn / (self.C_msn * self.A_m) + dphi_se_dClsn
        dphi_sn_dCasn = dq_sn_dCasn / (self.C_msn * self.A_m) + dphi_se_dCasn
        dphi_sn_dVsn = dphi_se_dVsn

        dphi_sn_dNadn = dphi_se_dNadn
        dphi_sn_dKdn = dphi_se_dKdn
        dphi_sn_dCldn = dphi_se_dCldn
        dphi_sn_dCadn = dphi_se_dCadn
        dphi_sn_dVdn = dphi_se_dVdn

        dphi_sn_dNasg = dphi_se_dNasg
        dphi_sn_dKsg = dphi_se_dKsg
        dphi_sn_dClsg = dphi_se_dClsg
        dphi_sn_dVsg = dphi_se_dVsg

        dphi_sn_dNadg = dphi_se_dNadg
        dphi_sn_dKdg = dphi_se_dKdg
        dphi_sn_dCldg = dphi_se_dCldg
        dphi_sn_dVdg = dphi_se_dVdg

        dphi_sn_dNase = dphi_se_dNase
        dphi_sn_dKse = dphi_se_dKse
        dphi_sn_dClse = dphi_se_dClse
        dphi_sn_dCase = dphi_se_dCase
        dphi_sn_dVse = dphi_se_dVse

        dphi_sn_dNade = dphi_se_dNade
        dphi_sn_dKde = dphi_se_dKde
        dphi_sn_dClde = dphi_se_dClde
        dphi_sn_dCade = dphi_se_dCade
        dphi_sn_dVde = dphi_se_dVde

        # phi_sg
        dphi_sg_dNasn = dphi_se_dNasn
        dphi_sg_dKsn = dphi_se_dKsn 
        dphi_sg_dClsn = dphi_se_dClsn
        dphi_sg_dCasn = dphi_se_dCasn
        dphi_sg_dVsn = dphi_se_dVsn

        dphi_sg_dNadn = dphi_se_dNadn
        dphi_sg_dKdn = dphi_se_dKdn
        dphi_sg_dCldn = dphi_se_dCldn
        dphi_sg_dCadn = dphi_se_dCadn
        dphi_sg_dVdn = dphi_se_dVdn

        dphi_sg_dNasg = dq_sg_dNasg / (self.C_msg * self.A_m) + dphi_se_dNasg
        dphi_sg_dKsg = dq_sg_dKsg / (self.C_msg * self.A_m) + dphi_se_dKsg
        dphi_sg_dClsg = dq_sg_dClsg / (self.C_msg * self.A_m) + dphi_se_dClsg
        dphi_sg_dVsg = dphi_se_dVsg

        dphi_sg_dNadg = dphi_se_dNadg
        dphi_sg_dKdg = dphi_se_dKdg
        dphi_sg_dCldg = dphi_se_dCldg
        dphi_sg_dVdg = dphi_se_dVdg

        dphi_sg_dNase = dphi_se_dNase
        dphi_sg_dKse = dphi_se_dKse
        dphi_sg_dClse = dphi_se_dClse
        dphi_sg_dCase = dphi_se_dCase
        dphi_sg_dVse = dphi_se_dVse

        dphi_sg_dNade = dphi_se_dNade
        dphi_sg_dKde = dphi_se_dKde
        dphi_sg_dClde = dphi_se_dClde
        dphi_sg_dCade = dphi_se_dCade
        dphi_sg_dVde = dphi_se_dVde

        # phi_msn
        dphi_msn_dNasn = dq_sn_dNasn / (self.C_msn * self.A_m)
        dphi_msn_dKsn = dq_sn_dKsn / (self.C_msn * self.A_m)
        dphi_msn_dClsn = dq_sn_dClsn / (self.C_msn * self.A_m)
        dphi_msn_dCasn = dq_sn_dCasn / (self.C_msn * self.A_m)

        # phi_msg
        dphi_msg_dNasg = dq_sg_dNasg / (self.C_msg * self.A_m)
        dphi_msg_dKsg = dq_sg_dKsg / (self.C_msg * self.A_m)
        dphi_msg_dClsg = dq_sg_dClsg / (self.C_msg * self.A_m)

        # phi_mdn
        dphi_mdn_dNadn = dphi_dn_dNadn
        dphi_mdn_dKdn = dphi_dn_dKdn 
        dphi_mdn_dCldn = dphi_dn_dCldn 
        dphi_mdn_dCadn = dphi_dn_dCadn 

        # phi_mdg
        dphi_mdg_dNadg = dphi_dg_dNadg
        dphi_mdg_dKdg = dphi_dg_dKdg 
        dphi_mdg_dCldg = dphi_dg_dCldg 


        # Current fluxes derivatives
        
        # j_Na_msn
        dj_Na_msn_dNasn = self.g_Na_leak_n / (self.F*self.Z_Na) * (dphi_msn_dNasn - dE_Na_sn_dNasn)  \
                    + 3*self.dj_pump_n_dn(self.cNa_sn, self.cK_se, dcNa_sn_dNasn) \
                    + self.dj_nkcc1_dn(self.cNa_sn, dcNa_sn_dNasn, self.cK_se ) \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * 2*self.m_inf(phi_msn) * self.dm_inf(phi_msn, dphi_msn_dNasn)  * (phi_msn - E_Na_sn)  \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * self.m_inf(phi_msn)**2  * (dphi_msn_dNasn - dE_Na_sn_dNasn)
        dj_Na_msn_dKsn = self.g_Na_leak_n / (self.F*self.Z_Na) * dphi_msn_dKsn  \
                    + self.dj_nkcc1_dn(self.cK_sn, dcK_sn_dKsn, self.cK_se ) \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * 2*self.m_inf(phi_msn) * self.dm_inf(phi_msn, dphi_msn_dKsn)  * (phi_msn - E_Na_sn)  \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * self.m_inf(phi_msn)**2  * dphi_msn_dKsn
        dj_Na_msn_dClsn = self.g_Na_leak_n / (self.F*self.Z_Na) * dphi_msn_dClsn  \
                    + self.dj_nkcc1_dn(self.cCl_sn, dcCl_sn_dClsn, self.cK_se ) \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * 2*self.m_inf(phi_msn) * self.dm_inf(phi_msn, dphi_msn_dClsn)  * (phi_msn - E_Na_sn )  \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * self.m_inf(phi_msn)**2  * dphi_msn_dClsn 
        dj_Na_msn_dCasn = self.g_Na_leak_n / (self.F*self.Z_Na) * dphi_msn_dCasn  \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * 2*self.m_inf(phi_msn) * self.dm_inf(phi_msn, dphi_msn_dCasn)  * (phi_msn - E_Na_sn) \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * self.m_inf(phi_msn)**2  * dphi_msn_dCasn  \
                    - 2*self.U_Cadec*dcCa_sn_dCasn*self.V_sn/self.A_m
        dj_Na_msn_dVsn = self.g_Na_leak_n / (self.F*self.Z_Na) * ( - dE_Na_sn_dVsn)  \
                    + 3*self.dj_pump_n_dn(self.cNa_sn, self.cK_se, dcNa_sn_dVsn) \
                    + self.dj_nkcc1_dVn(self.cNa_sn, self.cK_sn, self.cCl_sn, dcNa_sn_dVsn, dcK_sn_dVsn, dcCl_sn_dVsn, self.cK_se) \
                    - 2*self.U_Cadec*dcCa_sn_dVsn *self.V_sn/self.A_m \
                    - 2*self.U_Cadec*(self.cCa_sn - self.cbCa_sn)/self.A_m \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * self.m_inf(phi_msn)**2  * (- dE_Na_sn_dVsn)
        dj_Na_msn_dNase = self.g_Na_leak_n / (self.F*self.Z_Na) * (- dE_Na_sn_dNase)  \
                    + self.dj_nkcc1_de(self.cNa_se, dcNa_se_dNase, self.cK_se ) \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * self.m_inf(phi_msn)**2  * (- dE_Na_sn_dNase)
        dj_Na_msn_dKse = 3*self.dj_pump_n_de(self.cNa_sn, self.cK_se, dcK_se_dKse) \
                    + self.dj_nkcc1_dKe(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se, dcK_se_dKse)
        dj_Na_msn_dClse = self.dj_nkcc1_de(self.cCl_se, dcCl_se_dClse, self.cK_se )
        dj_Na_msn_dVse = self.g_Na_leak_n / (self.F*self.Z_Na) * (- dE_Na_sn_dVse)  \
                    + 3*self.dj_pump_n_de(self.cNa_sn, self.cK_se, dcK_se_dVse) \
                    + self.dj_nkcc1_dVe(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se, dcNa_se_dVse, dcK_se_dVse, dcCl_se_dVse) \
                    + self.g_Na * self.h / (self.F*self.Z_Na) * self.m_inf(phi_msn)**2  * (- dE_Na_sn_dVse)
        dj_Na_msn_dh =  self.g_Na * self.m_inf(phi_msn)**2 * (phi_msn - E_Na_sn) / (self.F*self.Z_Na)

        # j_K_msn
        dj_K_msn_dNasn = self.g_K_leak_n / (self.F*self.Z_K) * dphi_msn_dNasn  \
                    - 2*self.dj_pump_n_dn(self.cNa_sn, self.cK_se, dcNa_sn_dNasn) \
                    + self.dj_nkcc1_dn(self.cNa_sn, dcNa_sn_dNasn, self.cK_se ) \
                    + self.g_DR * self.n * dphi_msn_dNasn / (self.F*self.Z_K) 
        dj_K_msn_dKsn = self.g_K_leak_n / (self.F*self.Z_K) * (dphi_msn_dKsn - dE_K_sn_dKsn) \
                    + self.dj_kcc2_dn(self.cK_sn, dcK_sn_dKsn) \
                    + self.dj_nkcc1_dn(self.cK_sn, dcK_sn_dKsn, self.cK_se ) \
                    + self.g_DR * self.n * (dphi_msn_dKsn - dE_K_sn_dKsn) / (self.F*self.Z_K)
        dj_K_msn_dClsn = self.g_K_leak_n / (self.F*self.Z_K) * dphi_msn_dClsn  \
                    + self.dj_kcc2_dn(self.cCl_sn, dcCl_sn_dClsn) \
                    + self.dj_nkcc1_dn(self.cCl_sn, dcCl_sn_dClsn, self.cK_se ) \
                    + self.g_DR * self.n * dphi_msn_dClsn / (self.F*self.Z_K) 
        dj_K_msn_dCasn = self.g_K_leak_n / (self.F*self.Z_K) * dphi_msn_dCasn  \
                    + self.g_DR * self.n * dphi_msn_dCasn / (self.F*self.Z_K) 
        dj_K_msn_dVsn = self.g_K_leak_n / (self.F*self.Z_K) * (- dE_K_sn_dVsn) \
                    - 2*self.dj_pump_n_dn(self.cNa_sn, self.cK_se, dcNa_sn_dVsn) \
                    + self.dj_kcc2_dVn(self.cK_sn, self.cCl_sn, dcK_sn_dVsn, dcCl_sn_dVsn) \
                    + self.dj_nkcc1_dVn(self.cNa_sn, self.cK_sn, self.cCl_sn, dcNa_sn_dVsn, dcK_sn_dVsn, dcCl_sn_dVsn, self.cK_se) \
                    + self.g_DR * self.n * (- dE_K_sn_dVsn) / (self.F*self.Z_K)
        dj_K_msn_dNase = self.dj_nkcc1_de(self.cNa_se, dcNa_se_dNase, self.cK_se )
        dj_K_msn_dKse = self.g_K_leak_n / (self.F*self.Z_K) * (- dE_K_sn_dKse) \
                    - 2*self.dj_pump_n_de(self.cNa_sn, self.cK_se, dcK_se_dKse) \
                    + self.dj_kcc2_de(self.cK_se, dcK_se_dKse) \
                    + self.dj_nkcc1_dKe(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se, dcK_se_dKse) \
                    + self.g_DR * self.n * (- dE_K_sn_dKse) / (self.F*self.Z_K)
        dj_K_msn_dClse = self.dj_kcc2_de(self.cCl_se, dcCl_se_dClse) \
                    + self.dj_nkcc1_de(self.cCl_se, dcCl_se_dClse, self.cK_se )
        dj_K_msn_dVse = self.g_K_leak_n / (self.F*self.Z_K) * (- dE_K_sn_dVse) \
                    - 2*self.dj_pump_n_de(self.cNa_sn, self.cK_se, dcK_se_dVse) \
                    + self.dj_kcc2_dVe(self.cK_se, self.cCl_se, dcK_se_dVse, dcCl_se_dVse)  \
                    + self.dj_nkcc1_dVe(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se, dcNa_se_dVse, dcK_se_dVse, dcCl_se_dVse) \
                    + self.g_DR * self.n * (- dE_K_sn_dVse) / (self.F*self.Z_K)
        dj_K_msn_dn = self.g_DR * (phi_msn - E_K_sn) / (self.F*self.Z_K)

        # j_Cl_sn
        dj_Cl_msn_dNasn = self.g_Cl_leak_n / (self.F*self.Z_Cl) * dphi_msn_dNasn  \
                    + 2*self.dj_nkcc1_dn(self.cNa_sn, dcNa_sn_dNasn, self.cK_se )
        dj_Cl_msn_dKsn = self.g_Cl_leak_n / (self.F*self.Z_Cl) * dphi_msn_dKsn  \
                    + self.dj_kcc2_dn(self.cK_sn, dcK_sn_dKsn) \
                    + 2*self.dj_nkcc1_dn(self.cK_sn, dcK_sn_dKsn, self.cK_se )
        dj_Cl_msn_dClsn = self.g_Cl_leak_n / (self.F*self.Z_Cl) * (dphi_msn_dClsn - dE_Cl_sn_dClsn)  \
                    + self.dj_kcc2_dn(self.cCl_sn, dcCl_sn_dClsn) \
                    + 2*self.dj_nkcc1_dn(self.cCl_sn, dcCl_sn_dClsn, self.cK_se )
        dj_Cl_msn_dCasn = self.g_Cl_leak_n / (self.F*self.Z_Cl) * dphi_msn_dCasn 
        dj_Cl_msn_dVsn = self.g_Cl_leak_n / (self.F*self.Z_Cl) * (- dE_Cl_sn_dVsn)  \
                    + self.dj_kcc2_dVn(self.cK_sn, self.cCl_sn, dcK_sn_dVsn, dcCl_sn_dVsn) \
                    + 2*self.dj_nkcc1_dVn(self.cNa_sn, self.cK_sn, self.cCl_sn, dcNa_sn_dVsn, dcK_sn_dVsn, dcCl_sn_dVsn, self.cK_se) 
        dj_Cl_msn_dNase = 2*self.dj_nkcc1_de(self.cNa_se, dcNa_se_dNase, self.cK_se )
        dj_Cl_msn_dKse = self.dj_kcc2_de(self.cK_se, dcK_se_dKse) \
                    + 2*self.dj_nkcc1_dKe(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se, dcK_se_dKse) 
        dj_Cl_msn_dClse = self.g_Cl_leak_n / (self.F*self.Z_Cl) * (- dE_Cl_sn_dClse)  \
                    + self.dj_kcc2_de(self.cCl_se, dcCl_se_dClse) \
                    + 2*self.dj_nkcc1_de(self.cCl_se, dcCl_se_dClse, self.cK_se )
        dj_Cl_msn_dVse = self.g_Cl_leak_n / (self.F*self.Z_Cl) * (- dE_Cl_sn_dVse)  \
                    + self.dj_kcc2_dVe(self.cK_se, self.cCl_se, dcK_se_dVse, dcCl_se_dVse)  \
                    + 2*self.dj_nkcc1_dVe(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se, dcNa_se_dVse, dcK_se_dVse, dcCl_se_dVse)

        # j_Ca_sn
        dj_Ca_msn_dCasn = self.U_Cadec *self.V_sn/self.A_m * dcCa_sn_dCasn
        dj_Ca_msn_dVsn = self.U_Cadec * dcCa_sn_dVsn*self.V_sn/self.A_m \
                        + self.U_Cadec * (self.cCa_sn - self.cbCa_sn)/self.A_m

        # j_Na_mdn
        dj_Na_mdn_dNadn = self.g_Na_leak_n / (self.F*self.Z_Na)* (dphi_mdn_dNadn - dE_Na_dn_dNadn) \
                    + 3*self.dj_pump_n_dn(self.cNa_dn, self.cK_de, dcNa_dn_dNadn) \
                    + self.dj_nkcc1_dn (self.cNa_dn,dcNa_dn_dNadn, self.cK_de)
        dj_Na_mdn_dKdn = self.g_Na_leak_n / (self.F*self.Z_Na)* dphi_mdn_dKdn  \
                    + self.dj_nkcc1_dn (self.cK_dn,dcK_dn_dKdn, self.cK_de)
        dj_Na_mdn_dCldn = self.g_Na_leak_n / (self.F*self.Z_Na)* dphi_mdn_dCldn  \
                    + self.dj_nkcc1_dn (self.cCl_dn,dcCl_dn_dCldn, self.cK_de)
        dj_Na_mdn_dCadn = self.g_Na_leak_n / (self.F*self.Z_Na)* dphi_mdn_dCadn  \
                    - 2*self.U_Cadec*self.V_dn/self.A_m * dcCa_dn_dCadn
        dj_Na_mdn_dVdn = self.g_Na_leak_n / (self.F*self.Z_Na)* (- dE_Na_dn_dVdn) \
                    + 3*self.dj_pump_n_dn(self.cNa_dn, self.cK_de, dcNa_dn_dVdn) \
                    + self.dj_nkcc1_dVn (self.cNa_dn, self.cK_dn, self.cCl_dn, dcNa_dn_dVdn, dcK_dn_dVdn, dcCl_dn_dVdn, self.cK_de)  \
                    - 2*self.U_Cadec*self.V_dn/self.A_m * dcCa_dn_dVdn \
                    - 2*self.U_Cadec*(self.cCa_dn - self.cbCa_dn)/self.A_m
        dj_Na_mdn_dNade = - self.g_Na_leak_n / (self.F*self.Z_Na)* dE_Na_dn_dNade \
                    + self.dj_nkcc1_de (self.cNa_de,dcNa_de_dNade, self.cK_de)
        dj_Na_mdn_dKde = 3*self.dj_pump_n_de(self.cNa_dn, self.cK_de, dcK_de_dKde) \
                    + self.dj_nkcc1_dKe(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de, dcK_de_dKde) 
        dj_Na_mdn_dClde = self.dj_nkcc1_de (self.cCl_de,dcCl_de_dClde, self.cK_de)
        dj_Na_mdn_dVde = - self.g_Na_leak_n / (self.F*self.Z_Na)* dE_Na_dn_dVde \
                    + 3*self.dj_pump_n_de(self.cNa_dn, self.cK_de, dcK_de_dVde) \
                    + self.dj_nkcc1_dVe(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de, dcNa_de_dVde, dcK_de_dVde, dcCl_de_dVde) 

        # j_K_dn
        dj_K_mdn_dNadn = self.g_K_leak_n / (self.F*self.Z_K) * dphi_mdn_dNadn  \
                    - 2*self.dj_pump_n_dn(self.cNa_dn, self.cK_de, dcNa_dn_dNadn) \
                    + self.dj_nkcc1_dn(self.cNa_dn,dcNa_dn_dNadn, self.cK_de) \
                    + self.g_AHP * self.q / (self.F*self.Z_K) * dphi_mdn_dNadn  \
                    + self.g_C * self.c / (self.F*self.Z_K) * self.chi() * dphi_mdn_dNadn 
        dj_K_mdn_dKdn = self.g_K_leak_n / (self.F*self.Z_K) * (dphi_mdn_dKdn - dE_K_dn_dKdn) \
                    + self.dj_kcc2_dn(self.cK_dn, dcK_dn_dKdn) \
                    + self.dj_nkcc1_dn(self.cK_dn,dcK_dn_dKdn, self.cK_de) \
                    + self.g_AHP * self.q / (self.F*self.Z_K) * (dphi_mdn_dKdn - dE_K_dn_dKdn)  \
                    + self.g_C * self.c / (self.F*self.Z_K) * self.chi() * (dphi_mdn_dKdn - dE_K_dn_dKdn)
        dj_K_mdn_dCldn = self.g_K_leak_n / (self.F*self.Z_K) * dphi_mdn_dCldn  \
                    + self.dj_kcc2_dn(self.cCl_dn, dcCl_dn_dCldn) \
                    + self.dj_nkcc1_dn(self.cCl_dn,dcCl_dn_dCldn, self.cK_de) \
                    + self.g_AHP * self.q / (self.F*self.Z_K) * dphi_mdn_dCldn  \
                    + self.g_C * self.c / (self.F*self.Z_K) * self.chi() * dphi_mdn_dCldn 
        dj_K_mdn_dCadn = self.g_K_leak_n / (self.F*self.Z_K) * dphi_mdn_dCadn  \
                    + self.g_AHP * self.q / (self.F*self.Z_K) * dphi_mdn_dCadn  \
                    + self.g_C * self.c / (self.F*self.Z_K) * self.chi() * dphi_mdn_dCadn \
                    + self.g_C * self.c * self.dchi_dCa_dn() * (phi_mdn - E_K_dn) / (self.F*self.Z_K)
        dj_K_mdn_dVdn = self.g_K_leak_n / (self.F*self.Z_K) * (- dE_K_dn_dVdn) \
                    - 2*self.dj_pump_n_dn(self.cNa_dn, self.cK_de, dcNa_dn_dVdn) \
                    + self.dj_kcc2_dVn(self.cK_dn, self.cCl_dn, dcK_dn_dVdn, dcCl_dn_dVdn) \
                    + self.dj_nkcc1_dVn(self.cNa_dn, self.cK_dn, self.cCl_dn, dcNa_dn_dVdn, dcK_dn_dVdn, dcCl_dn_dVdn, self.cK_de)  \
                    + self.g_AHP * self.q / (self.F*self.Z_K) * (- dE_K_dn_dVdn)  \
                    + self.g_C * self.c / (self.F*self.Z_K) * self.chi() * (- dE_K_dn_dVdn) \
                    + self.g_C * self.c * self.dchi_dV_dn() * (phi_mdn - E_K_dn) / (self.F*self.Z_K)
        dj_K_mdn_dNade = self.dj_nkcc1_de (self.cNa_de,dcNa_de_dNade, self.cK_de)
        dj_K_mdn_dKde = - self.g_K_leak_n / (self.F*self.Z_K) * dE_K_dn_dKde  \
                    - 2*self.dj_pump_n_de(self.cNa_dn, self.cK_de, dcK_de_dKde) \
                    + self.dj_kcc2_de(self.cK_de, dcK_de_dKde) \
                    + self.dj_nkcc1_dKe(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de, dcK_de_dKde) \
                    - self.g_AHP * self.q / (self.F*self.Z_K) * dE_K_dn_dKde  \
                    - self.g_C * self.c / (self.F*self.Z_K) * self.chi() * dE_K_dn_dKde  
        dj_K_mdn_dClde = self.dj_kcc2_de(self.cCl_de, dcCl_de_dClde) \
                    + self.dj_nkcc1_de (self.cCl_de,dcCl_de_dClde, self.cK_de)
        dj_K_mdn_dVde = - self.g_K_leak_n / (self.F*self.Z_K) * dE_K_dn_dVde  \
                    - 2*self.dj_pump_n_de(self.cNa_dn, self.cK_de, dcK_de_dVde) \
                    + self.dj_kcc2_dVe(self.cK_de, self.cCl_de, dcK_de_dVde, dcCl_de_dVde) \
                    + self.dj_nkcc1_dVe(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de, dcNa_de_dVde, dcK_de_dVde, dcCl_de_dVde) \
                    - self.g_AHP * self.q / (self.F*self.Z_K) * dE_K_dn_dVde  \
                    - self.g_C * self.c / (self.F*self.Z_K) * self.chi() * dE_K_dn_dVde 
        dj_K_mdn_dq = self.g_AHP * (phi_mdn - E_K_dn) / (self.F*self.Z_K) 
        dj_K_mdn_dc = self.g_C * self.chi() * (phi_mdn - E_K_dn) / (self.F*self.Z_K)

        # j_Cl_dn
        dj_Cl_mdn_dNadn = self.g_Cl_leak_n / (self.F*self.Z_Cl) * dphi_mdn_dNadn  \
                    + 2*self.dj_nkcc1_dn (self.cNa_dn,dcNa_dn_dNadn, self.cK_de)
        dj_Cl_mdn_dKdn = self.g_Cl_leak_n / (self.F*self.Z_Cl) * dphi_mdn_dKdn  \
                    + self.dj_kcc2_dn(self.cK_dn, dcK_dn_dKdn) \
                    + 2*self.dj_nkcc1_dn (self.cK_dn,dcK_dn_dKdn, self.cK_de)
        dj_Cl_mdn_dCldn = self.g_Cl_leak_n / (self.F*self.Z_Cl)* (dphi_mdn_dCldn - dE_Cl_dn_dCldn) \
                    + self.dj_kcc2_dn(self.cCl_dn, dcCl_dn_dCldn) \
                    + 2*self.dj_nkcc1_dn (self.cCl_dn,dcCl_dn_dCldn, self.cK_de)
        dj_Cl_mdn_dCadn = self.g_Cl_leak_n / (self.F*self.Z_Cl)* dphi_mdn_dCadn 
        dj_Cl_mdn_dVdn = self.g_Cl_leak_n / (self.F*self.Z_Cl)* (- dE_Cl_dn_dVdn) \
                    + self.dj_kcc2_dVn(self.cK_dn, self.cCl_dn, dcK_dn_dVdn, dcCl_dn_dVdn) \
                    + 2*self.dj_nkcc1_dVn(self.cNa_dn, self.cK_dn, self.cCl_dn, dcNa_dn_dVdn, dcK_dn_dVdn, dcCl_dn_dVdn, self.cK_de) 
        dj_Cl_mdn_dNade = 2*self.dj_nkcc1_de (self.cNa_de,dcNa_de_dNade, self.cK_de)
        dj_Cl_mdn_dKde = self.dj_kcc2_de(self.cK_de, dcK_de_dKde) \
                    + 2*self.dj_nkcc1_dKe(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de, dcK_de_dKde) 
        dj_Cl_mdn_dClde = - self.g_Cl_leak_n / (self.F*self.Z_Cl) * dE_Cl_dn_dClde  \
                    + self.dj_kcc2_de(self.cCl_de, dcCl_de_dClde) \
                    + 2*self.dj_nkcc1_de (self.cCl_de,dcCl_de_dClde, self.cK_de)
        dj_Cl_mdn_dVde = - self.g_Cl_leak_n / (self.F*self.Z_Cl) * dE_Cl_dn_dVde  \
                    + self.dj_kcc2_dVe(self.cK_de, self.cCl_de, dcK_de_dVde, dcCl_de_dVde) \
                    + 2*self.dj_nkcc1_dVe(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de, dcNa_de_dVde, dcK_de_dVde, dcCl_de_dVde) 

        # j_Ca_dn
        dj_Ca_mdn_dNadn = self.g_Ca / (self.F*self.Z_Ca) * self.s**2 * self.z * dphi_mdn_dNadn             
        dj_Ca_mdn_dKdn = self.g_Ca / (self.F*self.Z_Ca) * self.s**2 * self.z * dphi_mdn_dKdn 
        dj_Ca_mdn_dCldn = self.g_Ca / (self.F*self.Z_Ca) * self.s**2 * self.z * dphi_mdn_dCldn 
        dj_Ca_mdn_dCadn = self.g_Ca / (self.F*self.Z_Ca) * self.s**2 * self.z * (dphi_mdn_dCadn - dE_Ca_dn_dCadn) \
                    + self.U_Cadec*dcCa_dn_dCadn*self.V_dn/self.A_m        
        dj_Ca_mdn_dVdn = self.g_Ca / (self.F*self.Z_Ca) * self.s**2 * self.z * (- dE_Ca_dn_dVdn) \
                    + self.U_Cadec*dcCa_dn_dVdn*self.V_dn/self.A_m \
                    + self.U_Cadec*(self.cCa_dn - self.cbCa_dn)/self.A_m  
        dj_Ca_mdn_dCade = - self.g_Ca / (self.F*self.Z_Ca) * self.s**2 * self.z * dE_Ca_dn_dCade
        dj_Ca_mdn_dVde = - self.g_Ca / (self.F*self.Z_Ca) * self.s**2 * self.z * dE_Ca_dn_dVde
        dj_Ca_mdn_ds = self.g_Ca * 2 * self.s * self.z * (phi_mdn - E_Ca_dn) / (self.F*self.Z_Ca)
        dj_Ca_mdn_dz = self.g_Ca * self.s**2 * (phi_mdn - E_Ca_dn) / (self.F*self.Z_Ca)

        # j_Na_sg
        dj_Na_msg_dNasg = self.g_Na_leak_g / self.F * dphi_msg_dNasg \
                    - self.g_Na_leak_g / self.F * dE_Na_sg_dNasg  \
                    + 3*self.dj_pump_g_dg(self.cNa_sg, self.cK_se, dcNa_sg_dNasg)
        dj_Na_msg_dKsg = self.g_Na_leak_g / self.F * dphi_msg_dKsg
        dj_Na_msg_dClsg = self.g_Na_leak_g / self.F * dphi_msg_dClsg 
        dj_Na_msg_dVsg = 3*self.dj_pump_g_dg(self.cNa_sg, self.cK_se, dcNa_sg_dVsg) \
                    - self.g_Na_leak_g / self.F * dE_Na_sg_dVsg  
        dj_Na_msg_dNase = - self.g_Na_leak_g / self.F * dE_Na_sg_dNase  
        dj_Na_msg_dKse = 3*self.dj_pump_g_de(self.cNa_sg, self.cK_se, dcK_se_dKse)
        dj_Na_msg_dVse = - self.g_Na_leak_g / self.F * dE_Na_sg_dVse  \
                    + 3*self.dj_pump_g_de(self.cNa_sg, self.cK_se, dcK_se_dVse)

        # helper function
        f_sqrt = np.sqrt(self.cK_se/self.cbK_se)

        # j_K_msg
        dj_K_msg_dNasg = self.g_K_IR / self.F * f_sqrt * self.df_dphi_m(phi_msg, E_K_sg, dphi_msg_dNasg, 's') * (phi_msg - E_K_sg)  \
                    + self.g_K_IR / self.F * f_sqrt * self.f(phi_msg, E_K_sg, 's') * dphi_msg_dNasg  \
                    - 2 * self.dj_pump_g_dg(self.cNa_sg, self.cK_se, dcNa_sg_dNasg)
        dj_K_msg_dKsg = self.g_K_IR / self.F * f_sqrt * self.df_dphi_m(phi_msg, E_K_sg, dphi_msg_dKsg, 's') * (phi_msg - E_K_sg) \
                    + self.g_K_IR / self.F * f_sqrt * self.df_dE_K(phi_msg, E_K_sg, dE_K_sg_dKsg, 's') * (phi_msg - E_K_sg)  \
                    + self.g_K_IR / self.F * f_sqrt * self.f(phi_msg, E_K_sg, 's') * dphi_msg_dKsg  \
                    - self.g_K_IR / self.F * f_sqrt * self.f(phi_msg, E_K_sg, 's') * dE_K_sg_dKsg  
        dj_K_msg_dClsg = self.g_K_IR / self.F * f_sqrt * self.df_dphi_m(phi_msg, E_K_sg, dphi_msg_dClsg, 's') * (phi_msg - E_K_sg) \
                    + self.g_K_IR / self.F * f_sqrt * self.f(phi_msg, E_K_sg, 's') * dphi_msg_dClsg   
        q1 =  self.g_K_IR / self.F * f_sqrt * self.df_dE_K(phi_msg, E_K_sg, dE_K_sg_dVsg, 's') * (phi_msg - E_K_sg)
        q2 = - 2*self.dj_pump_g_dg(self.cNa_sg, self.cK_se, dcNa_sg_dVsg)
        q3 = - self.g_K_IR / self.F * f_sqrt * self.f(phi_msg, E_K_sg, 's') * dE_K_sg_dVsg
        dj_K_msg_dVsg = + self.g_K_IR / self.F * f_sqrt * self.df_dE_K(phi_msg, E_K_sg, dE_K_sg_dVsg, 's') * (phi_msg - E_K_sg)  \
                    - 2*self.dj_pump_g_dg(self.cNa_sg, self.cK_se, dcNa_sg_dVsg) \
                    - self.g_K_IR / self.F * f_sqrt * self.f(phi_msg, E_K_sg, 's') * dE_K_sg_dVsg 
        dj_K_msg_dKse = self.g_K_IR / self.F * f_sqrt * self.df_dE_K(phi_msg, E_K_sg, dE_K_sg_dKse, 's') * (phi_msg - E_K_sg)  \
                    + self.g_K_IR / self.F * f_sqrt * self.f(phi_msg, E_K_sg, 's') * (- dE_K_sg_dKse) \
                    + self.g_K_IR / self.F * self.df_sqrt(self.cK_se, dcK_se_dKse, self.cbK_se) * self.f(phi_msg, E_K_sg, 's') * (phi_msg - E_K_sg) \
                    -2*self.dj_pump_g_de(self.cNa_sg, self.cK_se, dcK_se_dKse)
        dj_K_msg_dVse = self.g_K_IR / self.F * f_sqrt * self.df_dE_K(phi_msg, E_K_sg, dE_K_sg_dVse, 's') * (phi_msg - E_K_sg)  \
                    + self.g_K_IR / self.F * f_sqrt * self.f(phi_msg, E_K_sg, 's') * (- dE_K_sg_dVse) \
                    + self.g_K_IR / self.F * self.df_sqrt(self.cK_se, dcK_se_dVse, self.cbK_se) * self.f(phi_msg, E_K_sg, 's') * (phi_msg - E_K_sg)  \
                    - 2*self.dj_pump_g_de(self.cNa_sg, self.cK_se, dcK_se_dVse)

        # j_Cl_msg
        dj_Cl_msg_dNasg = - self.g_Cl_leak_g / self.F * dphi_msg_dNasg
        dj_Cl_msg_dKsg = - self.g_Cl_leak_g / self.F * dphi_msg_dKsg
        dj_Cl_msg_dClsg = - self.g_Cl_leak_g / self.F * (dphi_msg_dClsg - dE_Cl_sg_dClsg)
        dj_Cl_msg_dClse = - self.g_Cl_leak_g / self.F * (- dE_Cl_sg_dClse)
        dj_Cl_msg_dVsg = - self.g_Cl_leak_g / self.F * (- dE_Cl_sg_dVsg)
        dj_Cl_msg_dVse = - self.g_Cl_leak_g / self.F * (- dE_Cl_sg_dVse)

        # j_Na_dg
        dj_Na_mdg_dNadg = self.g_Na_leak_g / self.F * (dphi_mdg_dNadg - dE_Na_dg_dNadg) \
                    + 3*self.dj_pump_g_dg(self.cNa_dg, self.cK_de, dcNa_dg_dNadg)
        dj_Na_mdg_dKdg = self.g_Na_leak_g / self.F * dphi_mdg_dKdg 
        dj_Na_mdg_dCldg = self.g_Na_leak_g / self.F * dphi_mdg_dCldg 
        dj_Na_mdg_dVdg = self.g_Na_leak_g / self.F * (- dE_Na_dg_dVdg) \
                    + 3*self.dj_pump_g_dg(self.cNa_dg, self.cK_de, dcNa_dg_dVdg)
        dj_Na_mdg_dNade = - self.g_Na_leak_g / self.F * dE_Na_dg_dNade 
        dj_Na_mdg_dKde =  3*self.dj_pump_g_de(self.cNa_dg, self.cK_de, dcK_de_dKde)
        dj_Na_mdg_dVde = - self.g_Na_leak_g / self.F * dE_Na_dg_dVde \
                    + 3*self.dj_pump_g_de(self.cNa_dg, self.cK_de, dcK_de_dVde)

        # helper function
        f_sqrt_d = np.sqrt(self.cK_de/self.cbK_de)

        # j_K_dg
        dj_K_mdg_dNadg = self.g_K_IR / self.F * f_sqrt_d * self.df_dphi_m(phi_mdg, E_K_dg, dphi_mdg_dNadg, 'd') * (phi_mdg - E_K_dg)  \
                    + self.g_K_IR / self.F * f_sqrt_d * self.f(phi_mdg, E_K_dg, 'd') * dphi_mdg_dNadg  \
                    - 2*self.dj_pump_g_dg(self.cNa_dg, self.cK_de, dcNa_dg_dNadg)     
        dj_K_mdg_dKdg = self.g_K_IR / self.F * f_sqrt_d * self.df_dphi_m(phi_mdg, E_K_dg, dphi_mdg_dKdg, 'd') * (phi_mdg -  E_K_dg)  \
                    + self.g_K_IR / self.F * f_sqrt_d * self.df_dE_K(phi_mdg, E_K_dg, dE_K_dg_dKdg, 'd') * (phi_mdg -  E_K_dg)  \
                    + self.g_K_IR / self.F * f_sqrt_d * self.f(phi_mdg, E_K_dg, 'd') * (dphi_mdg_dKdg - dE_K_dg_dKdg)           
        dj_K_mdg_dCldg = self.g_K_IR / self.F * f_sqrt_d * self.df_dphi_m(phi_mdg, E_K_dg, dphi_mdg_dCldg, 'd') * (phi_mdg -E_K_dg) \
                    + self.g_K_IR / self.F * f_sqrt_d * self.f(phi_mdg, E_K_dg, 'd') * dphi_mdg_dCldg    
        dj_K_mdg_dVdg = - 2*self.dj_pump_g_dg(self.cNa_dg, self.cK_de, dcNa_dg_dVdg) \
                    + self.g_K_IR / self.F * f_sqrt_d * self.df_dE_K(phi_mdg, E_K_dg, dE_K_dg_dVdg, 'd') * (phi_mdg -  E_K_dg)  \
                    + self.g_K_IR / self.F * f_sqrt_d * self.f(phi_mdg, E_K_dg, 'd') * (- dE_K_dg_dVdg) 
        dj_K_mdg_dKde = self.g_K_IR / self.F * f_sqrt_d * self.df_dE_K(phi_mdg, E_K_dg, dE_K_dg_dKde, 'd') * (phi_mdg -E_K_dg)  \
                    - self.g_K_IR / self.F * f_sqrt_d * self.f(phi_mdg, E_K_dg, 'd') * dE_K_dg_dKde  \
                    + self.g_K_IR / self.F * self.df_sqrt(self.cK_de, dcK_de_dKde, self.cbK_de) * self.f(phi_mdg, E_K_dg, 'd') * (phi_mdg -E_K_dg) \
                    - 2*self.dj_pump_g_de(self.cNa_dg, self.cK_de, dcK_de_dKde) 
        dj_K_mdg_dVde = self.g_K_IR / self.F * f_sqrt_d * self.df_dE_K(phi_mdg, E_K_dg, dE_K_dg_dVde, 'd') * (phi_mdg -E_K_dg)  \
                    - self.g_K_IR / self.F * f_sqrt_d * self.f(phi_mdg, E_K_dg, 'd') * dE_K_dg_dVde  \
                    + self.g_K_IR / self.F * self.df_sqrt(self.cK_de, dcK_de_dVde, self.cbK_de) * self.f(phi_mdg, E_K_dg, 'd') * (phi_mdg -E_K_dg) \
                    - 2*self.dj_pump_g_de(self.cNa_dg, self.cK_de, dcK_de_dVde)  

        # j_Cl_dg
        dj_Cl_mdg_dNadg = - self.g_Cl_leak_g * dphi_mdg_dNadg / self.F
        dj_Cl_mdg_dKdg = - self.g_Cl_leak_g * dphi_mdg_dKdg / self.F
        dj_Cl_mdg_dCldg = - self.g_Cl_leak_g * (dphi_mdg_dCldg - dE_Cl_dg_dCldg) / self.F
        dj_Cl_mdg_dClde = self.g_Cl_leak_g * dE_Cl_dg_dClde / self.F

        dj_Cl_mdg_dVdg = - self.g_Cl_leak_g * (- dE_Cl_dg_dVdg) / self.F
        dj_Cl_mdg_dVde = self.g_Cl_leak_g * dE_Cl_dg_dVde / self.F

        # j_Na_in
        dj_Na_in_dNasn = self.dj_k_diff(self.D_Na, self.lamda_i, -dcNa_sn_dNasn) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_i, phi_sn, phi_dn, dcNa_sn_dNasn) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dNasn)
        dj_Na_in_dKsn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dKsn)
        dj_Na_in_dClsn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dClsn)
        dj_Na_in_dCasn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dCasn)
        dj_Na_in_dVsn = self.dj_k_diff(self.D_Na, self.lamda_i, -dcNa_sn_dVsn) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_i, phi_sn, phi_dn, dcNa_sn_dVsn) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dVsn)
        dj_Na_in_dNadn = self.dj_k_diff(self.D_Na, self.lamda_i, dcNa_dn_dNadn) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_i, phi_sn, phi_dn, dcNa_dn_dNadn) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dNadn) \
            + self.dj_k_drift_dphi_d(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_dn_dNadn)
        dj_Na_in_dKdn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dKdn)\
            + self.dj_k_drift_dphi_d(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_dn_dKdn)
        dj_Na_in_dCldn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dCldn)\
            + self.dj_k_drift_dphi_d(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_dn_dCldn)
        dj_Na_in_dCadn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dCadn)\
            + self.dj_k_drift_dphi_d(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_dn_dCadn)
        dj_Na_in_dVdn = self.dj_k_diff(self.D_Na, self.lamda_i, dcNa_dn_dVdn) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_i, phi_sn, phi_dn, dcNa_dn_dVdn) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dVdn)
        dj_Na_in_dNasg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dNasg)
        dj_Na_in_dKsg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dKsg)
        dj_Na_in_dClsg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dClsg)
        dj_Na_in_dVsg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dVsg)
        dj_Na_in_dNadg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dNadg)
        dj_Na_in_dKdg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dKdg)
        dj_Na_in_dCldg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dCldg)
        dj_Na_in_dVdg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dVdg)
        dj_Na_in_dNase = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dNase)
        dj_Na_in_dKse = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dKse)
        dj_Na_in_dClse = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dClse)
        dj_Na_in_dCase = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dCase)
        dj_Na_in_dVse = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dVse)
        dj_Na_in_dNade = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dNade)
        dj_Na_in_dKde = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dKde)
        dj_Na_in_dClde = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dClde)
        dj_Na_in_dCade = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dCade)
        dj_Na_in_dVde = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn , dphi_sn_dVde)

        # j_K_in
        dj_K_in_dNasn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dNasn)
        dj_K_in_dKsn = self.dj_k_diff(self.D_K, self.lamda_i, -dcK_sn_dKsn) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_i, phi_sn, phi_dn, dcK_sn_dKsn) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dKsn)
        dj_K_in_dClsn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dClsn)
        dj_K_in_dCasn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dCasn)
        dj_K_in_dVsn = self.dj_k_diff(self.D_K, self.lamda_i, -dcK_sn_dVsn) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_i, phi_sn, phi_dn, dcK_sn_dVsn) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dVsn)
        dj_K_in_dNadn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dNadn) \
            + self.dj_k_drift_dphi_d(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_dn_dNadn)
        dj_K_in_dKdn = self.dj_k_diff(self.D_K, self.lamda_i, dcK_dn_dKdn) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_i, phi_sn, phi_dn, dcK_dn_dKdn) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dKdn) \
            + self.dj_k_drift_dphi_d(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_dn_dKdn)
        dj_K_in_dCldn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dCldn) \
            + self.dj_k_drift_dphi_d(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_dn_dCldn)
        dj_K_in_dCadn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dCadn) \
            + self.dj_k_drift_dphi_d(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_dn_dCadn)
        dj_K_in_dVdn = self.dj_k_diff(self.D_K, self.lamda_i, dcK_dn_dVdn) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_i, phi_sn, phi_dn, dcK_dn_dVdn) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dVdn)
        dj_K_in_dNasg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dNasg)
        dj_K_in_dKsg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dKsg)
        dj_K_in_dClsg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dClsg)
        dj_K_in_dVsg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dVsg)
        dj_K_in_dNadg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dNadg)
        dj_K_in_dKdg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dKdg)
        dj_K_in_dCldg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dCldg)
        dj_K_in_dVdg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dVdg)
        dj_K_in_dNase = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dNase)
        dj_K_in_dKse = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dKse)
        dj_K_in_dClse = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dClse)
        dj_K_in_dCase = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dCase)
        dj_K_in_dVse = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dVse)
        dj_K_in_dNade = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dNade)
        dj_K_in_dKde = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dKde)
        dj_K_in_dClde = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dClde)
        dj_K_in_dCade = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dCade)
        dj_K_in_dVde = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn , dphi_sn_dVde)

        # j_Cl_in
        dj_Cl_in_dNasn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dNasn)
        dj_Cl_in_dKsn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dKsn)
        dj_Cl_in_dClsn = self.dj_k_diff(self.D_Cl, self.lamda_i, -dcCl_sn_dClsn) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_i, phi_sn, phi_dn, dcCl_sn_dClsn) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dClsn)
        dj_Cl_in_dCasn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dCasn)
        dj_Cl_in_dVsn = self.dj_k_diff(self.D_Cl, self.lamda_i, -dcCl_sn_dVsn) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_i, phi_sn, phi_dn, dcCl_sn_dVsn) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dVsn)
        dj_Cl_in_dNadn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dNadn) \
            + self.dj_k_drift_dphi_d(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_dn_dNadn)
        dj_Cl_in_dKdn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dKdn) \
            + self.dj_k_drift_dphi_d(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_dn_dKdn)
        dj_Cl_in_dCldn = self.dj_k_diff(self.D_Cl, self.lamda_i, dcCl_dn_dCldn) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_i, phi_sn, phi_dn, dcCl_dn_dCldn) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dCldn) \
            + self.dj_k_drift_dphi_d(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_dn_dCldn)
        dj_Cl_in_dCadn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dCadn) \
            + self.dj_k_drift_dphi_d(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_dn_dCadn)
        dj_Cl_in_dVdn = self.dj_k_diff(self.D_Cl, self.lamda_i, dcCl_dn_dVdn) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_i, phi_sn, phi_dn, dcCl_dn_dVdn) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dVdn)
        dj_Cl_in_dNasg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dNasg)
        dj_Cl_in_dKsg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dKsg)
        dj_Cl_in_dClsg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dClsg)
        dj_Cl_in_dVsg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dVsg)
        dj_Cl_in_dNadg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dNadg)
        dj_Cl_in_dKdg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dKdg)
        dj_Cl_in_dCldg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dCldg)
        dj_Cl_in_dVdg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn, dphi_sn_dVdg)
        dj_Cl_in_dNase = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dNase)
        dj_Cl_in_dKse = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn, dphi_sn_dKse)
        dj_Cl_in_dClse = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dClse)
        dj_Cl_in_dCase = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn, dphi_sn_dCase)
        dj_Cl_in_dVse = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dVse)
        dj_Cl_in_dNade = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dNade)
        dj_Cl_in_dKde = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn, dphi_sn_dKde)
        dj_Cl_in_dClde = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dClde)
        dj_Cl_in_dCade = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dCade)
        dj_Cl_in_dVde = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn , dphi_sn_dVde)

        # j_Ca_in
        dj_Ca_in_dNasn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dNasn)
        dj_Ca_in_dKsn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dKsn)
        dj_Ca_in_dClsn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dClsn)
        dj_Ca_in_dCasn = self.dj_k_diff(self.D_Ca, self.lamda_i, -dfree_cCa_sn_dCasn) \
            + self.dj_k_drift_dck(self.D_Ca, self.Z_Ca, self.lamda_i, phi_sn, phi_dn, dfree_cCa_sn_dCasn) \
            + self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dCasn)
        dj_Ca_in_dVsn = self.dj_k_diff(self.D_Ca, self.lamda_i, -dfree_cCa_sn_dVsn) \
            + self.dj_k_drift_dck(self.D_Ca, self.Z_Ca, self.lamda_i, phi_sn, phi_dn, dfree_cCa_sn_dVsn) \
            + self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dVsn)
        dj_Ca_in_dNadn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dNadn) \
            + self.dj_k_drift_dphi_d(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_dn_dNadn)
        dj_Ca_in_dKdn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dKdn) \
            + self.dj_k_drift_dphi_d(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_dn_dKdn)
        dj_Ca_in_dCldn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dCldn) \
            + self.dj_k_drift_dphi_d(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_dn_dCldn)
        dj_Ca_in_dCadn = self.dj_k_diff(self.D_Ca, self.lamda_i, dfree_cCa_dn_dCadn) \
            + self.dj_k_drift_dck(self.D_Ca, self.Z_Ca, self.lamda_i, phi_sn, phi_dn, dfree_cCa_dn_dCadn) \
            + self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dCadn) \
            + self.dj_k_drift_dphi_d(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_dn_dCadn)
        dj_Ca_in_dVdn = self.dj_k_diff(self.D_Ca, self.lamda_i, dfree_cCa_dn_dVdn) \
            + self.dj_k_drift_dck(self.D_Ca, self.Z_Ca, self.lamda_i, phi_sn, phi_dn, dfree_cCa_dn_dVdn) \
            + self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dVdn)
        dj_Ca_in_dNasg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dNasg)
        dj_Ca_in_dKsg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dKsg)
        dj_Ca_in_dClsg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dClsg)
        dj_Ca_in_dVsg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dVsg)
        dj_Ca_in_dNadg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dNadg)
        dj_Ca_in_dKdg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dKdg)
        dj_Ca_in_dCldg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dCldg)
        dj_Ca_in_dVdg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn, dphi_sn_dVdg)
        dj_Ca_in_dNase = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dNase)
        dj_Ca_in_dKse = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn, dphi_sn_dKse)
        dj_Ca_in_dClse = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dClse)
        dj_Ca_in_dCase = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn, dphi_sn_dCase)
        dj_Ca_in_dVse = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dVse)
        dj_Ca_in_dNade = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dNade)
        dj_Ca_in_dKde = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn, dphi_sn_dKde)
        dj_Ca_in_dClde = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dClde)
        dj_Ca_in_dCade = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dCade)
        dj_Ca_in_dVde = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn , dphi_sn_dVde)

        # j_Na_ig
        dj_Na_ig_dNasn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dNasn)
        dj_Na_ig_dKsn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, dphi_sg_dKsn)
        dj_Na_ig_dClsn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dClsn)
        dj_Na_ig_dCasn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, dphi_sg_dCasn)
        dj_Na_ig_dVsn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dVsn)
        dj_Na_ig_dNadn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dNadn)
        dj_Na_ig_dKdn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, dphi_sg_dKdn)
        dj_Na_ig_dCldn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dCldn)
        dj_Na_ig_dCadn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dCadn)
        dj_Na_ig_dVdn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dVdn)
        dj_Na_ig_dNasg = self.dj_k_diff(self.D_Na, self.lamda_i, -dcNa_sg_dNasg) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_i, phi_sg, phi_dg, dcNa_sg_dNasg) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dNasg) 
        dj_Na_ig_dKsg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, dphi_sg_dKsg)
        dj_Na_ig_dClsg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dClsg)
        dj_Na_ig_dVsg = self.dj_k_diff(self.D_Na, self.lamda_i, -dcNa_sg_dVsg) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_i, phi_sg, phi_dg, dcNa_sg_dVsg) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dVsg)
        dj_Na_ig_dNadg = self.dj_k_diff(self.D_Na, self.lamda_i, dcNa_dg_dNadg) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_i, phi_sg, phi_dg, dcNa_dg_dNadg) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dNadg) \
            + self.dj_k_drift_dphi_d(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_dg_dNadg) 
        dj_Na_ig_dKdg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, dphi_sg_dKdg) \
            + self.dj_k_drift_dphi_d(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_dg_dKdg) 
        dj_Na_ig_dCldg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dCldg) \
            + self.dj_k_drift_dphi_d(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_dg_dCldg) 
        dj_Na_ig_dVdg = self.dj_k_diff(self.D_Na, self.lamda_i, dcNa_dg_dVdg) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_i, phi_sg, phi_dg, dcNa_dg_dVdg) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dVdg)
        dj_Na_ig_dNase = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dNase)
        dj_Na_ig_dKse = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, dphi_sg_dKse)
        dj_Na_ig_dClse = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dClse)
        dj_Na_ig_dCase = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, dphi_sg_dCase)
        dj_Na_ig_dVse = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dVse)
        dj_Na_ig_dNade = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dNade)
        dj_Na_ig_dKde = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, dphi_sg_dKde)
        dj_Na_ig_dClde = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dClde)
        dj_Na_ig_dCade = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dCade)
        dj_Na_ig_dVde = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg , dphi_sg_dVde)

        # j_K_ig
        dj_K_ig_dNasn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dNasn)
        dj_K_ig_dKsn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg, dphi_sg_dKsn)
        dj_K_ig_dClsn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dClsn)
        dj_K_ig_dCasn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg, dphi_sg_dCasn)
        dj_K_ig_dVsn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dVsn)
        dj_K_ig_dNadn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dNadn)
        dj_K_ig_dKdn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg, dphi_sg_dKdn)
        dj_K_ig_dCldn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dCldn)
        dj_K_ig_dCadn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dCadn)
        dj_K_ig_dVdn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dVdn)
        dj_K_ig_dNasg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dNasg) 
        dj_K_ig_dKsg = self.dj_k_diff(self.D_K, self.lamda_i, -dcK_sg_dKsg) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_i, phi_sg, phi_dg, dcK_sg_dKsg) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg, dphi_sg_dKsg)
        dj_K_ig_dClsg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dClsg)
        dj_K_ig_dVsg = self.dj_k_diff(self.D_K, self.lamda_i, -dcK_sg_dVsg) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_i, phi_sg, phi_dg, dcK_sg_dVsg) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dVsg)
        dj_K_ig_dNadg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dNadg) \
            + self.dj_k_drift_dphi_d(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_dg_dNadg) 
        dj_K_ig_dKdg = self.dj_k_diff(self.D_K, self.lamda_i, dcK_dg_dKdg) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_i, phi_sg, phi_dg, dcK_dg_dKdg) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i,self.cK_sg, self.cK_dg, dphi_sg_dKdg) \
            + self.dj_k_drift_dphi_d(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_dg_dKdg) 
        dj_K_ig_dCldg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dCldg) \
            + self.dj_k_drift_dphi_d(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_dg_dCldg) 
        dj_K_ig_dVdg = self.dj_k_diff(self.D_K, self.lamda_i, dcK_dg_dVdg) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_i, phi_sg, phi_dg, dcK_dg_dVdg) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dVdg)
        dj_K_ig_dNase = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dNase)
        dj_K_ig_dKse = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg, dphi_sg_dKse)
        dj_K_ig_dClse = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dClse)
        dj_K_ig_dCase = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg, dphi_sg_dCase)
        dj_K_ig_dVse = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dVse)
        dj_K_ig_dNade = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dNade)
        dj_K_ig_dKde = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg, dphi_sg_dKde)
        dj_K_ig_dClde = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dClde)
        dj_K_ig_dCade = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dCade)
        dj_K_ig_dVde = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg , dphi_sg_dVde)

        # j_Cl_ig
        dj_Cl_ig_dNasn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dNasn)
        dj_Cl_ig_dKsn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg, dphi_sg_dKsn)
        dj_Cl_ig_dClsn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dClsn)
        dj_Cl_ig_dCasn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg, dphi_sg_dCasn)
        dj_Cl_ig_dVsn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dVsn)
        dj_Cl_ig_dNadn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dNadn)
        dj_Cl_ig_dKdn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg, dphi_sg_dKdn)
        dj_Cl_ig_dCldn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dCldn)
        dj_Cl_ig_dCadn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dCadn)
        dj_Cl_ig_dVdn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dVdn)
        dj_Cl_ig_dNasg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dNasg) 
        dj_Cl_ig_dKsg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg, dphi_sg_dKsg)
        dj_Cl_ig_dClsg = self.dj_k_diff(self.D_Cl, self.lamda_i, -dcCl_sg_dClsg) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_i, phi_sg, phi_dg, dcCl_sg_dClsg) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dClsg)
        dj_Cl_ig_dVsg = self.dj_k_diff(self.D_Cl, self.lamda_i, -dcCl_sg_dVsg) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_i, phi_sg, phi_dg, dcCl_sg_dVsg) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dVsg)
        dj_Cl_ig_dNadg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dNadg) \
            + self.dj_k_drift_dphi_d(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_dg_dNadg) 
        dj_Cl_ig_dKdg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i,self.cCl_sg, self.cCl_dg, dphi_sg_dKdg) \
            + self.dj_k_drift_dphi_d(self.D_Cl, self.Z_Cl, self.lamda_i,self.cCl_sg, self.cCl_dg , dphi_dg_dKdg) 
        dj_Cl_ig_dCldg = self.dj_k_diff(self.D_Cl, self.lamda_i, dcCl_dg_dCldg) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_i, phi_sg, phi_dg, dcCl_dg_dCldg) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i,self.cCl_sg, self.cCl_dg , dphi_sg_dCldg) \
            + self.dj_k_drift_dphi_d(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_dg_dCldg) 
        dj_Cl_ig_dVdg = self.dj_k_diff(self.D_Cl, self.lamda_i, dcCl_dg_dVdg) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_i, phi_sg, phi_dg, dcCl_dg_dVdg) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dVdg)
        dj_Cl_ig_dNase = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dNase)
        dj_Cl_ig_dKse = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg, dphi_sg_dKse)
        dj_Cl_ig_dClse = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dClse)
        dj_Cl_ig_dCase = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg, dphi_sg_dCase)
        dj_Cl_ig_dVse = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dVse)
        dj_Cl_ig_dNade = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dNade)
        dj_Cl_ig_dKde = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg, dphi_sg_dKde)
        dj_Cl_ig_dClde = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dClde)
        dj_Cl_ig_dCade = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dCade)
        dj_Cl_ig_dVde = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg , dphi_sg_dVde)

        # j_Na_e
        dj_Na_e_dNasn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dNasn)
        dj_Na_e_dKsn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, dphi_se_dKsn)
        dj_Na_e_dClsn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dClsn)
        dj_Na_e_dCasn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, dphi_se_dCasn)
        dj_Na_e_dVsn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dVsn)
        dj_Na_e_dNadn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dNadn)
        dj_Na_e_dKdn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, dphi_se_dKdn)
        dj_Na_e_dCldn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dCldn)
        dj_Na_e_dCadn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dCadn)
        dj_Na_e_dVdn = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dVdn)
        dj_Na_e_dNasg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dNasg)
        dj_Na_e_dKsg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, dphi_se_dKsg)
        dj_Na_e_dClsg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dClsg)
        dj_Na_e_dVsg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dVsg)
        dj_Na_e_dNadg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dNadg)
        dj_Na_e_dKdg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, dphi_se_dKdg)
        dj_Na_e_dCldg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dCldg)
        dj_Na_e_dVdg = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dVdg)
        dj_Na_e_dNase = self.dj_k_diff(self.D_Na, self.lamda_e, -dcNa_se_dNase) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_e,  phi_se, phi_de, dcNa_se_dNase) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dNase)
        dj_Na_e_dKse = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, dphi_se_dKse)
        dj_Na_e_dClse = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dClse)
        dj_Na_e_dCase = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, dphi_se_dCase)
        dj_Na_e_dVse = self.dj_k_diff(self.D_Na, self.lamda_e, -dcNa_se_dVse) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_e,  phi_se, phi_de, dcNa_se_dVse) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dVse)
        dj_Na_e_dNade = self.dj_k_diff(self.D_Na, self.lamda_e, dcNa_de_dNade) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_e,  phi_se, phi_de, dcNa_de_dNade) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dNade)
        dj_Na_e_dKde = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, dphi_se_dKde)
        dj_Na_e_dClde = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dClde)
        dj_Na_e_dCade = self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dCade)
        dj_Na_e_dVde = self.dj_k_diff(self.D_Na, self.lamda_e, dcNa_de_dVde) \
            + self.dj_k_drift_dck(self.D_Na, self.Z_Na, self.lamda_e,  phi_se, phi_de, dcNa_de_dVde) \
            + self.dj_k_drift_dphi_s(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de , dphi_se_dVde)
        
        # j_K_e
        dj_K_e_dNasn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dNasn)
        dj_K_e_dKsn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, dphi_se_dKsn)
        dj_K_e_dClsn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dClsn)
        dj_K_e_dCasn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, dphi_se_dCasn)
        dj_K_e_dVsn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dVsn)
        dj_K_e_dNadn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dNadn)
        dj_K_e_dKdn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, dphi_se_dKdn)
        dj_K_e_dCldn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dCldn)
        dj_K_e_dCadn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dCadn)
        dj_K_e_dVdn = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dVdn)
        dj_K_e_dNasg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dNasg)
        dj_K_e_dKsg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, dphi_se_dKsg)
        dj_K_e_dClsg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dClsg)
        dj_K_e_dVsg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dVsg)
        dj_K_e_dNadg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dNadg)
        dj_K_e_dKdg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, dphi_se_dKdg)
        dj_K_e_dCldg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dCldg)
        dj_K_e_dVdg = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dVdg)
        dj_K_e_dNase = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dNase)
        dj_K_e_dKse = self.dj_k_diff(self.D_K, self.lamda_e, -dcK_se_dKse) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_e,  phi_se, phi_de, dcK_se_dKse) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, dphi_se_dKse)
        dj_K_e_dClse = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dClse)
        dj_K_e_dCase = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e,self.cK_se, self.cK_de, dphi_se_dCase)
        dj_K_e_dVse = self.dj_k_diff(self.D_K, self.lamda_e, -dcK_se_dVse) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_e,  phi_se, phi_de, dcK_se_dVse) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dVse)
        dj_K_e_dNade = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dNade)
        dj_K_e_dKde = self.dj_k_diff(self.D_K, self.lamda_e, dcK_de_dKde) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_e,  phi_se, phi_de, dcK_de_dKde) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, dphi_se_dKde)
        dj_K_e_dClde = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dClde)
        dj_K_e_dCade = self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de , dphi_se_dCade)
        dj_K_e_dVde = self.dj_k_diff(self.D_K, self.lamda_e, dcK_de_dVde) \
            + self.dj_k_drift_dck(self.D_K, self.Z_K, self.lamda_e,  phi_se, phi_de, dcK_de_dVde) \
            + self.dj_k_drift_dphi_s(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, dphi_se_dVde)

        # j_Cl_e
        dj_Cl_e_dNasn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dNasn)
        dj_Cl_e_dKsn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, dphi_se_dKsn)
        dj_Cl_e_dClsn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dClsn)
        dj_Cl_e_dCasn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, dphi_se_dCasn)
        dj_Cl_e_dVsn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dVsn)
        dj_Cl_e_dNadn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dNadn)
        dj_Cl_e_dKdn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, dphi_se_dKdn)
        dj_Cl_e_dCldn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dCldn)
        dj_Cl_e_dCadn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dCadn)
        dj_Cl_e_dVdn = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dVdn)
        dj_Cl_e_dNasg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dNasg)
        dj_Cl_e_dKsg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, dphi_se_dKsg)
        dj_Cl_e_dClsg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dClsg)
        dj_Cl_e_dVsg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, dphi_se_dVsg)
        dj_Cl_e_dNadg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dNadg)
        dj_Cl_e_dKdg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, dphi_se_dKdg)
        dj_Cl_e_dCldg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dCldg)
        dj_Cl_e_dVdg = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dVdg)
        dj_Cl_e_dNase = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dNase)
        dj_Cl_e_dKse = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, dphi_se_dKse)
        dj_Cl_e_dClse = self.dj_k_diff(self.D_Cl, self.lamda_e, -dcCl_se_dClse) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_e,  phi_se, phi_de, dcCl_se_dClse) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dClse)
        dj_Cl_e_dCase = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e,self.cCl_se, self.cCl_de, dphi_se_dCase)
        dj_Cl_e_dVse = self.dj_k_diff(self.D_Cl, self.lamda_e, -dcCl_se_dVse) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_e,  phi_se, phi_de, dcCl_se_dVse) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dVse)
        dj_Cl_e_dNade = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dNade)
        dj_Cl_e_dKde = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, dphi_se_dKde)
        dj_Cl_e_dClde = self.dj_k_diff(self.D_Cl, self.lamda_e, dcCl_de_dClde) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_e,  phi_se, phi_de, dcCl_de_dClde) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dClde)
        dj_Cl_e_dCade = self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de , dphi_se_dCade)
        dj_Cl_e_dVde = self.dj_k_diff(self.D_Cl, self.lamda_e, dcCl_de_dVde) \
            + self.dj_k_drift_dck(self.D_Cl, self.Z_Cl, self.lamda_e,  phi_se, phi_de, dcCl_de_dVde) \
            + self.dj_k_drift_dphi_s(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, dphi_se_dVde)

        # j_Ca_e
        dj_Ca_e_dNasn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dNasn)
        dj_Ca_e_dKsn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dKsn)
        dj_Ca_e_dClsn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dClsn)
        dj_Ca_e_dCasn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dCasn)
        dj_Ca_e_dVsn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dVsn)
        dj_Ca_e_dNadn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dNadn)
        dj_Ca_e_dKdn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dKdn)
        dj_Ca_e_dCldn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dCldn)
        dj_Ca_e_dCadn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dCadn)
        dj_Ca_e_dVdn = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dVdn)
        dj_Ca_e_dNasg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dNasg)
        dj_Ca_e_dKsg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dKsg)
        dj_Ca_e_dClsg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dClsg)
        dj_Ca_e_dVsg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dVsg)
        dj_Ca_e_dNadg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dNadg)
        dj_Ca_e_dKdg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dKdg)
        dj_Ca_e_dCldg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dCldg)
        dj_Ca_e_dVdg = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dVdg)
        dj_Ca_e_dNase = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dNase)
        dj_Ca_e_dKse = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dKse)
        dj_Ca_e_dClse = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dClse)
        dj_Ca_e_dCase = self.dj_k_diff(self.D_Ca, self.lamda_e, -dcCa_se_dCase) \
            + self.dj_k_drift_dck(self.D_Ca, self.Z_Ca, self.lamda_e,  phi_se, phi_de, dcCa_se_dCase) \
            + self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e,self.cCa_se, self.cCa_de, dphi_se_dCase)
        dj_Ca_e_dVse = self.dj_k_diff(self.D_Ca, self.lamda_e, -dcCa_se_dVse) \
            + self.dj_k_drift_dck(self.D_Ca, self.Z_Ca, self.lamda_e,  phi_se, phi_de, dcCa_se_dVse) \
            + self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dVse)
        dj_Ca_e_dNade = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dNade)
        dj_Ca_e_dKde = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dKde)
        dj_Ca_e_dClde = self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dClde)
        dj_Ca_e_dCade = self.dj_k_diff(self.D_Ca, self.lamda_e, dcCa_de_dCade) \
            + self.dj_k_drift_dck(self.D_Ca, self.Z_Ca, self.lamda_e,  phi_se, phi_de, dcCa_de_dCade) \
            + self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de , dphi_se_dCade)
        dj_Ca_e_dVde = self.dj_k_diff(self.D_Ca, self.lamda_e, dcCa_de_dVde) \
            + self.dj_k_drift_dck(self.D_Ca, self.Z_Ca, self.lamda_e,  phi_se, phi_de, dcCa_de_dVde) \
            + self.dj_k_drift_dphi_s(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, dphi_se_dVde)
        
        # Number of ions derivatives

        ''' dNadt_sn = -j_Na_msn*self.A_m - j_Na_in*self.A_i'''
        dNadt_sn_dNasn = -dj_Na_msn_dNasn*self.A_m - dj_Na_in_dNasn*self.A_i
        dNadt_sn_dKsn = -dj_Na_msn_dKsn*self.A_m - dj_Na_in_dKsn*self.A_i
        dNadt_sn_dClsn = -dj_Na_msn_dClsn*self.A_m - dj_Na_in_dClsn*self.A_i
        dNadt_sn_dCasn = -dj_Na_msn_dCasn*self.A_m - dj_Na_in_dCasn*self.A_i
        dNadt_sn_dVsn = -dj_Na_msn_dVsn*self.A_m - dj_Na_in_dVsn*self.A_i
        dNadt_sn_dNadn = - dj_Na_in_dNadn*self.A_i
        dNadt_sn_dKdn = - dj_Na_in_dKdn*self.A_i
        dNadt_sn_dCldn = - dj_Na_in_dCldn*self.A_i
        dNadt_sn_dCadn = - dj_Na_in_dCadn*self.A_i
        dNadt_sn_dVdn = - dj_Na_in_dVdn*self.A_i
        dNadt_sn_dNasg = - dj_Na_in_dNasg*self.A_i
        dNadt_sn_dKsg = - dj_Na_in_dKsg*self.A_i
        dNadt_sn_dClsg = - dj_Na_in_dClsg*self.A_i
        dNadt_sn_dVsg = - dj_Na_in_dVsg*self.A_i
        dNadt_sn_dNadg = - dj_Na_in_dNadg*self.A_i
        dNadt_sn_dKdg = - dj_Na_in_dKdg*self.A_i
        dNadt_sn_dCldg = - dj_Na_in_dCldg*self.A_i
        dNadt_sn_dVdg = - dj_Na_in_dVdg*self.A_i
        dNadt_sn_dNase = -dj_Na_msn_dNase*self.A_m - dj_Na_in_dNase*self.A_i
        dNadt_sn_dKse = -dj_Na_msn_dKse*self.A_m - dj_Na_in_dKse*self.A_i
        dNadt_sn_dClse = -dj_Na_msn_dClse*self.A_m - dj_Na_in_dClse*self.A_i
        dNadt_sn_dCase = - dj_Na_in_dCase*self.A_i
        dNadt_sn_dVse = -dj_Na_msn_dVse*self.A_m - dj_Na_in_dVse*self.A_i
        dNadt_sn_dNade = - dj_Na_in_dNade*self.A_i
        dNadt_sn_dKde = - dj_Na_in_dKde*self.A_i
        dNadt_sn_dClde = - dj_Na_in_dClde*self.A_i
        dNadt_sn_dCade = - dj_Na_in_dCade*self.A_i
        dNadt_sn_dVde = - dj_Na_in_dVde*self.A_i

        dNadt_sn_dh = -dj_Na_msn_dh*self.A_m

        '''dNadt_se = j_Na_msn*self.A_m + j_Na_msg*self.A_m - j_Na_e*self.A_e '''
        dNadt_se_dNasn = dj_Na_msn_dNasn*self.A_m - dj_Na_e_dNasn*self.A_e
        dNadt_se_dKsn = dj_Na_msn_dKsn*self.A_m - dj_Na_e_dKsn*self.A_e
        dNadt_se_dClsn = dj_Na_msn_dClsn*self.A_m - dj_Na_e_dClsn*self.A_e
        dNadt_se_dCasn = dj_Na_msn_dCasn*self.A_m - dj_Na_e_dCasn*self.A_e
        dNadt_se_dVsn = dj_Na_msn_dVsn*self.A_m - dj_Na_e_dVsn*self.A_e
        dNadt_se_dNadn = - dj_Na_e_dNadn*self.A_e
        dNadt_se_dKdn = - dj_Na_e_dKdn*self.A_e
        dNadt_se_dCldn = - dj_Na_e_dCldn*self.A_e
        dNadt_se_dCadn = - dj_Na_e_dCadn*self.A_e
        dNadt_se_dVdn = - dj_Na_e_dVdn*self.A_e
        dNadt_se_dNasg = dj_Na_msg_dNasg*self.A_m - dj_Na_e_dNasg*self.A_e
        dNadt_se_dKsg = dj_Na_msg_dKsg*self.A_m - dj_Na_e_dKsg*self.A_e
        dNadt_se_dClsg = dj_Na_msg_dClsg*self.A_m - dj_Na_e_dClsg*self.A_e
        dNadt_se_dVsg = dj_Na_msg_dVsg*self.A_m - dj_Na_e_dVsg*self.A_e
        dNadt_se_dNadg = - dj_Na_e_dNadg*self.A_e
        dNadt_se_dKdg = - dj_Na_e_dKdg*self.A_e
        dNadt_se_dCldg = - dj_Na_e_dCldg*self.A_e
        dNadt_se_dVdg = - dj_Na_e_dVdg*self.A_e
        dNadt_se_dNase = dj_Na_msn_dNase*self.A_m + dj_Na_msg_dNase*self.A_m - dj_Na_e_dNase*self.A_e
        dNadt_se_dKse = dj_Na_msn_dKse*self.A_m + dj_Na_msg_dKse*self.A_m - dj_Na_e_dKse*self.A_e
        dNadt_se_dClse = dj_Na_msn_dClse*self.A_m - dj_Na_e_dClse*self.A_e
        dNadt_se_dCase = - dj_Na_e_dCase*self.A_e
        dNadt_se_dVse = dj_Na_msn_dVse*self.A_m + dj_Na_msg_dVse*self.A_m - dj_Na_e_dVse*self.A_e
        dNadt_se_dNade = - dj_Na_e_dNade*self.A_e
        dNadt_se_dKde = - dj_Na_e_dKde*self.A_e
        dNadt_se_dClde = - dj_Na_e_dClde*self.A_e
        dNadt_se_dCade = - dj_Na_e_dCade*self.A_e
        dNadt_se_dVde = - dj_Na_e_dVde*self.A_e

        dNadt_se_dh = dj_Na_msn_dh*self.A_m

        '''dNadt_sg = -j_Na_msg*self.A_m - j_Na_ig*self.A_i'''
        dNadt_sg_dNasn = - dj_Na_ig_dNasn*self.A_i
        dNadt_sg_dKsn =  - dj_Na_ig_dKsn*self.A_i
        dNadt_sg_dClsn =  - dj_Na_ig_dClsn*self.A_i
        dNadt_sg_dCasn =  - dj_Na_ig_dCasn*self.A_i
        dNadt_sg_dVsn =  - dj_Na_ig_dVsn*self.A_i
        dNadt_sg_dNadn = - dj_Na_ig_dNadn*self.A_i
        dNadt_sg_dKdn =  - dj_Na_ig_dKdn*self.A_i
        dNadt_sg_dCldn = - dj_Na_ig_dCldn*self.A_i
        dNadt_sg_dCadn =  - dj_Na_ig_dCadn*self.A_i
        dNadt_sg_dVdn =  - dj_Na_ig_dVdn*self.A_i
        dNadt_sg_dNasg =  - dj_Na_msg_dNasg*self.A_m - dj_Na_ig_dNasg*self.A_i
        dNadt_sg_dKsg = - dj_Na_msg_dKsg*self.A_m - dj_Na_ig_dKsg*self.A_i
        dNadt_sg_dClsg =  - dj_Na_msg_dClsg*self.A_m - dj_Na_ig_dClsg*self.A_i
        dNadt_sg_dVsg =  - dj_Na_msg_dVsg*self.A_m - dj_Na_ig_dVsg*self.A_i
        dNadt_sg_dNadg =   - dj_Na_ig_dNadg*self.A_i
        dNadt_sg_dKdg = - dj_Na_ig_dKdg*self.A_i
        dNadt_sg_dCldg =  - dj_Na_ig_dCldg*self.A_i
        dNadt_sg_dVdg = - dj_Na_ig_dVdg*self.A_i
        dNadt_sg_dNase = - dj_Na_msg_dNase*self.A_m - dj_Na_ig_dNase*self.A_i
        dNadt_sg_dKse = - dj_Na_msg_dKse*self.A_m - dj_Na_ig_dKse*self.A_i
        dNadt_sg_dClse = - dj_Na_ig_dClse*self.A_i
        dNadt_sg_dCase = - dj_Na_ig_dCase*self.A_i
        dNadt_sg_dVse = - dj_Na_msg_dVse*self.A_m - dj_Na_ig_dVse*self.A_i
        dNadt_sg_dNade = - dj_Na_ig_dNade*self.A_i
        dNadt_sg_dKde = - dj_Na_ig_dKde*self.A_i
        dNadt_sg_dClde = - dj_Na_ig_dClde*self.A_i
        dNadt_sg_dCade = - dj_Na_ig_dCade*self.A_i
        dNadt_sg_dVde = - dj_Na_ig_dVde*self.A_i

        '''dNadt_dn = -j_Na_mdn*self.A_m + j_Na_in*self.A_i'''
        dNadt_dn_dNasn =  dj_Na_in_dNasn*self.A_i
        dNadt_dn_dKsn = dj_Na_in_dKsn*self.A_i
        dNadt_dn_dClsn =  dj_Na_in_dClsn*self.A_i
        dNadt_dn_dCasn = dj_Na_in_dCasn*self.A_i
        dNadt_dn_dVsn = dj_Na_in_dVsn*self.A_i
        dNadt_dn_dNadn = -dj_Na_mdn_dNadn*self.A_m + dj_Na_in_dNadn*self.A_i
        dNadt_dn_dKdn =  -dj_Na_mdn_dKdn*self.A_m + dj_Na_in_dKdn*self.A_i
        dNadt_dn_dCldn = -dj_Na_mdn_dCldn*self.A_m + dj_Na_in_dCldn*self.A_i
        dNadt_dn_dCadn = -dj_Na_mdn_dCadn*self.A_m + dj_Na_in_dCadn*self.A_i
        dNadt_dn_dVdn = -dj_Na_mdn_dVdn*self.A_m + dj_Na_in_dVdn*self.A_i
        dNadt_dn_dNasg = dj_Na_in_dNasg*self.A_i
        dNadt_dn_dKsg = dj_Na_in_dKsg*self.A_i
        dNadt_dn_dClsg = dj_Na_in_dClsg*self.A_i
        dNadt_dn_dVsg = dj_Na_in_dVsg*self.A_i
        dNadt_dn_dNadg = dj_Na_in_dNadg*self.A_i
        dNadt_dn_dKdg = dj_Na_in_dKdg*self.A_i
        dNadt_dn_dCldg = dj_Na_in_dCldg*self.A_i
        dNadt_dn_dVdg = dj_Na_in_dVdg*self.A_i
        dNadt_dn_dNase = dj_Na_in_dNase*self.A_i
        dNadt_dn_dKse = dj_Na_in_dKse*self.A_i
        dNadt_dn_dClse = dj_Na_in_dClse*self.A_i
        dNadt_dn_dCase = dj_Na_in_dCase*self.A_i
        dNadt_dn_dVse = dj_Na_in_dVse*self.A_i
        dNadt_dn_dNade = -dj_Na_mdn_dNade*self.A_m + dj_Na_in_dNade*self.A_i
        dNadt_dn_dKde = -dj_Na_mdn_dKde*self.A_m + dj_Na_in_dKde*self.A_i
        dNadt_dn_dClde = -dj_Na_mdn_dClde*self.A_m + dj_Na_in_dClde*self.A_i
        dNadt_dn_dCade = dj_Na_in_dCade*self.A_i
        dNadt_dn_dVde = -dj_Na_mdn_dVde*self.A_m + dj_Na_in_dVde*self.A_i

        '''dNadt_de = j_Na_mdn*self.A_m + j_Na_mdg*self.A_m + j_Na_e*self.A_e '''
        dNadt_de_dNasn = dj_Na_e_dNasn*self.A_e
        dNadt_de_dKsn =  dj_Na_e_dKsn*self.A_e
        dNadt_de_dClsn =  dj_Na_e_dClsn*self.A_e
        dNadt_de_dCasn =  dj_Na_e_dCasn*self.A_e
        dNadt_de_dVsn =  dj_Na_e_dVsn*self.A_e
        dNadt_de_dNadn =  dj_Na_mdn_dNadn*self.A_m + dj_Na_e_dNadn*self.A_e
        dNadt_de_dKdn =  dj_Na_mdn_dKdn*self.A_m + dj_Na_e_dKdn*self.A_e
        dNadt_de_dCldn =  dj_Na_mdn_dCldn*self.A_m + dj_Na_e_dCldn*self.A_e
        dNadt_de_dCadn =  dj_Na_mdn_dCadn*self.A_m + dj_Na_e_dCadn*self.A_e
        dNadt_de_dVdn =  dj_Na_mdn_dVdn*self.A_m + dj_Na_e_dVdn*self.A_e
        dNadt_de_dNasg =  dj_Na_e_dNasg*self.A_e
        dNadt_de_dKsg =  dj_Na_e_dKsg*self.A_e
        dNadt_de_dClsg =  dj_Na_e_dClsg*self.A_e
        dNadt_de_dVsg =  dj_Na_e_dVsg*self.A_e
        dNadt_de_dNadg =  dj_Na_mdg_dNadg*self.A_m + dj_Na_e_dNadg*self.A_e
        dNadt_de_dKdg = dj_Na_mdg_dKdg*self.A_m + dj_Na_e_dKdg*self.A_e
        dNadt_de_dCldg =  dj_Na_mdg_dCldg*self.A_m + dj_Na_e_dCldg*self.A_e
        dNadt_de_dVdg =  dj_Na_mdg_dVdg*self.A_m + dj_Na_e_dVdg*self.A_e
        dNadt_de_dNase =  dj_Na_e_dNase*self.A_e
        dNadt_de_dKse =  dj_Na_e_dKse*self.A_e
        dNadt_de_dClse =  dj_Na_e_dClse*self.A_e
        dNadt_de_dCase =  dj_Na_e_dCase*self.A_e
        dNadt_de_dVse =  dj_Na_e_dVse*self.A_e
        dNadt_de_dNade =  dj_Na_mdn_dNade*self.A_m + dj_Na_mdg_dNade*self.A_m + dj_Na_e_dNade*self.A_e
        dNadt_de_dKde =  dj_Na_mdn_dKde*self.A_m + dj_Na_mdg_dKde*self.A_m + dj_Na_e_dKde*self.A_e
        dNadt_de_dClde =  dj_Na_mdn_dClde*self.A_m + dj_Na_e_dClde*self.A_e
        dNadt_de_dCade =  dj_Na_e_dCade*self.A_e
        dNadt_de_dVde =  dj_Na_mdn_dVde*self.A_m + dj_Na_mdg_dVde*self.A_m + dj_Na_e_dVde*self.A_e

        '''dNadt_dg = -j_Na_mdg*self.A_m + j_Na_ig*self.A_i'''
        dNadt_dg_dNasn = dj_Na_ig_dNasn*self.A_i
        dNadt_dg_dKsn = dj_Na_ig_dKsn*self.A_i
        dNadt_dg_dClsn = dj_Na_ig_dClsn*self.A_i
        dNadt_dg_dCasn = dj_Na_ig_dCasn*self.A_i
        dNadt_dg_dVsn =  dj_Na_ig_dVsn*self.A_i
        dNadt_dg_dNadn = dj_Na_ig_dNadn*self.A_i
        dNadt_dg_dKdn =  dj_Na_ig_dKdn*self.A_i
        dNadt_dg_dCldn = dj_Na_ig_dCldn*self.A_i
        dNadt_dg_dCadn = dj_Na_ig_dCadn*self.A_i
        dNadt_dg_dVdn =  dj_Na_ig_dVdn*self.A_i
        dNadt_dg_dNasg = dj_Na_ig_dNasg*self.A_i
        dNadt_dg_dKsg = dj_Na_ig_dKsg*self.A_i
        dNadt_dg_dClsg = dj_Na_ig_dClsg*self.A_i
        dNadt_dg_dVsg = dj_Na_ig_dVsg*self.A_i
        dNadt_dg_dNadg = - dj_Na_mdg_dNadg*self.A_m + dj_Na_ig_dNadg*self.A_i
        dNadt_dg_dKdg = - dj_Na_mdg_dKdg*self.A_m + dj_Na_ig_dKdg*self.A_i
        dNadt_dg_dCldg =  - dj_Na_mdg_dCldg*self.A_m + dj_Na_ig_dCldg*self.A_i
        dNadt_dg_dVdg = - dj_Na_mdg_dVdg*self.A_m + dj_Na_ig_dVdg*self.A_i
        dNadt_dg_dNase = dj_Na_ig_dNase*self.A_i
        dNadt_dg_dKse = dj_Na_ig_dKse*self.A_i
        dNadt_dg_dClse = dj_Na_ig_dClse*self.A_i
        dNadt_dg_dCase = dj_Na_ig_dCase*self.A_i
        dNadt_dg_dVse = dj_Na_ig_dVse*self.A_i
        dNadt_dg_dNade = - dj_Na_mdg_dNade*self.A_m + dj_Na_ig_dNade*self.A_i
        dNadt_dg_dKde =  - dj_Na_mdg_dKde*self.A_m + dj_Na_ig_dKde*self.A_i
        dNadt_dg_dClde = dj_Na_ig_dClde*self.A_i
        dNadt_dg_dCade = dj_Na_ig_dCade*self.A_i
        dNadt_dg_dVde = - dj_Na_mdg_dVde*self.A_m + dj_Na_ig_dVde*self.A_i

        '''dKdt_sn = -j_K_msn*self.A_m - j_K_in*self.A_i'''
        dKdt_sn_dNasn = -dj_K_msn_dNasn*self.A_m - dj_K_in_dNasn*self.A_i
        dKdt_sn_dKsn = -dj_K_msn_dKsn*self.A_m - dj_K_in_dKsn*self.A_i
        dKdt_sn_dClsn = -dj_K_msn_dClsn*self.A_m - dj_K_in_dClsn*self.A_i
        dKdt_sn_dCasn = -dj_K_msn_dCasn*self.A_m - dj_K_in_dCasn*self.A_i
        dKdt_sn_dVsn = -dj_K_msn_dVsn*self.A_m - dj_K_in_dVsn*self.A_i
        dKdt_sn_dNadn = - dj_K_in_dNadn*self.A_i
        dKdt_sn_dKdn = - dj_K_in_dKdn*self.A_i
        dKdt_sn_dCldn = - dj_K_in_dCldn*self.A_i
        dKdt_sn_dCadn = - dj_K_in_dCadn*self.A_i
        dKdt_sn_dVdn = - dj_K_in_dVdn*self.A_i
        dKdt_sn_dNasg = - dj_K_in_dNasg*self.A_i
        dKdt_sn_dKsg = - dj_K_in_dKsg*self.A_i
        dKdt_sn_dClsg = - dj_K_in_dClsg*self.A_i
        dKdt_sn_dVsg = - dj_K_in_dVsg*self.A_i
        dKdt_sn_dNadg =- dj_K_in_dNadg*self.A_i
        dKdt_sn_dKdg = - dj_K_in_dKdg*self.A_i
        dKdt_sn_dCldg = - dj_K_in_dCldg*self.A_i
        dKdt_sn_dVdg = - dj_K_in_dVdg*self.A_i
        dKdt_sn_dNase = -dj_K_msn_dNase*self.A_m - dj_K_in_dNase*self.A_i
        dKdt_sn_dKse = -dj_K_msn_dKse*self.A_m - dj_K_in_dKse*self.A_i
        dKdt_sn_dClse = -dj_K_msn_dClse*self.A_m - dj_K_in_dClse*self.A_i
        dKdt_sn_dCase = - dj_K_in_dCase*self.A_i
        dKdt_sn_dVse = -dj_K_msn_dVse*self.A_m - dj_K_in_dVse*self.A_i
        dKdt_sn_dNade = - dj_K_in_dNade*self.A_i
        dKdt_sn_dKde = - dj_K_in_dKde*self.A_i
        dKdt_sn_dClde = - dj_K_in_dClde*self.A_i
        dKdt_sn_dCade = - dj_K_in_dCade*self.A_i
        dKdt_sn_dVde = - dj_K_in_dVde*self.A_i

        dKdt_sn_dn = -dj_K_msn_dn*self.A_m

        '''dKdt_se = j_K_msn*self.A_m + j_K_msg*self.A_m - j_K_e*self.A_e'''
        dKdt_se_dNasn = dj_K_msn_dNasn*self.A_m - dj_K_e_dNasn*self.A_e
        dKdt_se_dKsn = dj_K_msn_dKsn*self.A_m - dj_K_e_dKsn*self.A_e
        dKdt_se_dClsn = dj_K_msn_dClsn*self.A_m - dj_K_e_dClsn*self.A_e
        dKdt_se_dCasn = dj_K_msn_dCasn*self.A_m - dj_K_e_dCasn*self.A_e
        dKdt_se_dVsn = dj_K_msn_dVsn*self.A_m - dj_K_e_dVsn*self.A_e
        dKdt_se_dNadn = - dj_K_e_dNadn*self.A_e
        dKdt_se_dKdn = - dj_K_e_dKdn*self.A_e
        dKdt_se_dCldn = - dj_K_e_dCldn*self.A_e
        dKdt_se_dCadn = - dj_K_e_dCadn*self.A_e
        dKdt_se_dVdn = - dj_K_e_dVdn*self.A_e
        dKdt_se_dNasg = dj_K_msg_dNasg*self.A_m - dj_K_e_dNasg*self.A_e
        dKdt_se_dKsg = dj_K_msg_dKsg*self.A_m - dj_K_e_dKsg*self.A_e
        dKdt_se_dClsg = dj_K_msg_dClsg*self.A_m - dj_K_e_dClsg*self.A_e
        dKdt_se_dVsg = dj_K_msg_dVsg*self.A_m - dj_K_e_dVsg*self.A_e
        dKdt_se_dNadg = - dj_K_e_dNadg*self.A_e
        dKdt_se_dKdg = - dj_K_e_dKdg*self.A_e
        dKdt_se_dCldg = - dj_K_e_dCldg*self.A_e
        dKdt_se_dVdg = - dj_K_e_dVdg*self.A_e
        dKdt_se_dNase = dj_K_msn_dNase*self.A_m - dj_K_e_dNase*self.A_e
        dKdt_se_dKse = dj_K_msn_dKse*self.A_m + dj_K_msg_dKse*self.A_m - dj_K_e_dKse*self.A_e
        dKdt_se_dClse = dj_K_msn_dClse*self.A_m - dj_K_e_dClse*self.A_e
        dKdt_se_dCase = - dj_K_e_dCase*self.A_e
        dKdt_se_dVse = dj_K_msn_dVse*self.A_m + dj_K_msg_dVse*self.A_m - dj_K_e_dVse*self.A_e
        dKdt_se_dNade = - dj_K_e_dNade*self.A_e
        dKdt_se_dKde = - dj_K_e_dKde*self.A_e
        dKdt_se_dClde = - dj_K_e_dClde*self.A_e
        dKdt_se_dCade = - dj_K_e_dCade*self.A_e
        dKdt_se_dVde = - dj_K_e_dVde*self.A_e

        dKdt_se_dn = dj_K_msn_dn*self.A_m

        '''dKdt_sg = -j_K_msg*self.A_m - j_K_ig*self.A_i'''
        dKdt_sg_dNasn = - dj_K_ig_dNasn*self.A_i
        dKdt_sg_dKsn =  - dj_K_ig_dKsn*self.A_i
        dKdt_sg_dClsn =  - dj_K_ig_dClsn*self.A_i
        dKdt_sg_dCasn =  - dj_K_ig_dCasn*self.A_i
        dKdt_sg_dVsn =  - dj_K_ig_dVsn*self.A_i
        dKdt_sg_dNadn = - dj_K_ig_dNadn*self.A_i
        dKdt_sg_dKdn =  - dj_K_ig_dKdn*self.A_i
        dKdt_sg_dCldn = - dj_K_ig_dCldn*self.A_i
        dKdt_sg_dCadn = - dj_K_ig_dCadn*self.A_i
        dKdt_sg_dVdn =  - dj_K_ig_dVdn*self.A_i
        dKdt_sg_dNasg =  - dj_K_msg_dNasg*self.A_m - dj_K_ig_dNasg*self.A_i
        dKdt_sg_dKsg = - dj_K_msg_dKsg*self.A_m - dj_K_ig_dKsg*self.A_i
        dKdt_sg_dClsg =  - dj_K_msg_dClsg*self.A_m - dj_K_ig_dClsg*self.A_i
        dKdt_sg_dVsg =  - dj_K_msg_dVsg*self.A_m - dj_K_ig_dVsg*self.A_i
        dKdt_sg_dNadg =  - dj_K_ig_dNadg*self.A_i
        dKdt_sg_dKdg = - dj_K_ig_dKdg*self.A_i
        dKdt_sg_dCldg =  - dj_K_ig_dCldg*self.A_i
        dKdt_sg_dVdg = - dj_K_ig_dVdg*self.A_i
        dKdt_sg_dNase = - dj_K_ig_dNase*self.A_i
        dKdt_sg_dKse = - dj_K_msg_dKse*self.A_m - dj_K_ig_dKse*self.A_i
        dKdt_sg_dClse = - dj_K_ig_dClse*self.A_i
        dKdt_sg_dCase =- dj_K_ig_dCase*self.A_i
        dKdt_sg_dVse = - dj_K_msg_dVse*self.A_m - dj_K_ig_dVse*self.A_i
        dKdt_sg_dNade = - dj_K_ig_dNade*self.A_i
        dKdt_sg_dKde =  - dj_K_ig_dKde*self.A_i
        dKdt_sg_dClde = - dj_K_ig_dClde*self.A_i
        dKdt_sg_dCade = - dj_K_ig_dCade*self.A_i
        dKdt_sg_dVde = - dj_K_ig_dVde*self.A_i

        '''dKdt_dn = -j_K_mdn*self.A_m + j_K_in*self.A_i'''
        dKdt_dn_dNasn =  dj_K_in_dNasn*self.A_i
        dKdt_dn_dKsn = dj_K_in_dKsn*self.A_i
        dKdt_dn_dClsn =  dj_K_in_dClsn*self.A_i
        dKdt_dn_dCasn = dj_K_in_dCasn*self.A_i
        dKdt_dn_dVsn = dj_K_in_dVsn*self.A_i
        dKdt_dn_dNadn = -dj_K_mdn_dNadn*self.A_m + dj_K_in_dNadn*self.A_i
        dKdt_dn_dKdn =  -dj_K_mdn_dKdn*self.A_m + dj_K_in_dKdn*self.A_i
        dKdt_dn_dCldn = -dj_K_mdn_dCldn*self.A_m + dj_K_in_dCldn*self.A_i
        dKdt_dn_dCadn = -dj_K_mdn_dCadn*self.A_m + dj_K_in_dCadn*self.A_i
        dKdt_dn_dVdn = -dj_K_mdn_dVdn*self.A_m + dj_K_in_dVdn*self.A_i
        dKdt_dn_dNasg = dj_K_in_dNasg*self.A_i
        dKdt_dn_dKsg = dj_K_in_dKsg*self.A_i
        dKdt_dn_dClsg = dj_K_in_dClsg*self.A_i
        dKdt_dn_dVsg = dj_K_in_dVsg*self.A_i
        dKdt_dn_dNadg = dj_K_in_dNadg*self.A_i
        dKdt_dn_dKdg = dj_K_in_dKdg*self.A_i
        dKdt_dn_dCldg = dj_K_in_dCldg*self.A_i
        dKdt_dn_dVdg = dj_K_in_dVdg*self.A_i
        dKdt_dn_dNase = dj_K_in_dNase*self.A_i
        dKdt_dn_dKse = dj_K_in_dKse*self.A_i
        dKdt_dn_dClse = dj_K_in_dClse*self.A_i
        dKdt_dn_dCase = dj_K_in_dCase*self.A_i
        dKdt_dn_dVse = dj_K_in_dVse*self.A_i
        dKdt_dn_dNade = -dj_K_mdn_dNade*self.A_m + dj_K_in_dNade*self.A_i
        dKdt_dn_dKde = -dj_K_mdn_dKde*self.A_m + dj_K_in_dKde*self.A_i
        dKdt_dn_dClde = -dj_K_mdn_dClde*self.A_m + dj_K_in_dClde*self.A_i
        dKdt_dn_dCade = dj_K_in_dCade*self.A_i
        dKdt_dn_dVde = -dj_K_mdn_dVde*self.A_m + dj_K_in_dVde*self.A_i

        dKdt_dn_dq = -dj_K_mdn_dq*self.A_m
        dKdt_dn_dc = -dj_K_mdn_dc*self.A_m

        '''dKdt_de = j_K_mdn*self.A_m + j_K_mdg*self.A_m + j_K_e*self.A_e'''
        dKdt_de_dNasn = dj_K_e_dNasn*self.A_e
        dKdt_de_dKsn =  dj_K_e_dKsn*self.A_e
        dKdt_de_dClsn =  dj_K_e_dClsn*self.A_e
        dKdt_de_dCasn =  dj_K_e_dCasn*self.A_e
        dKdt_de_dVsn =  dj_K_e_dVsn*self.A_e
        dKdt_de_dNadn =  dj_K_mdn_dNadn*self.A_m + dj_K_e_dNadn*self.A_e
        dKdt_de_dKdn =  dj_K_mdn_dKdn*self.A_m + dj_K_e_dKdn*self.A_e
        dKdt_de_dCldn =  dj_K_mdn_dCldn*self.A_m + dj_K_e_dCldn*self.A_e
        dKdt_de_dCadn =  dj_K_mdn_dCadn*self.A_m + dj_K_e_dCadn*self.A_e
        dKdt_de_dVdn =  dj_K_mdn_dVdn*self.A_m + dj_K_e_dVdn*self.A_e
        dKdt_de_dNasg =  dj_K_e_dNasg*self.A_e
        dKdt_de_dKsg =  dj_K_e_dKsg*self.A_e
        dKdt_de_dClsg =  dj_K_e_dClsg*self.A_e
        dKdt_de_dVsg =  dj_K_e_dVsg*self.A_e
        dKdt_de_dNadg =  dj_K_mdg_dNadg*self.A_m + dj_K_e_dNadg*self.A_e
        dKdt_de_dKdg = dj_K_mdg_dKdg*self.A_m + dj_K_e_dKdg*self.A_e
        dKdt_de_dCldg =  dj_K_mdg_dCldg*self.A_m + dj_K_e_dCldg*self.A_e
        dKdt_de_dVdg =  dj_K_mdg_dVdg*self.A_m + dj_K_e_dVdg*self.A_e
        dKdt_de_dNase =  dj_K_e_dNase*self.A_e
        dKdt_de_dKse =  dj_K_e_dKse*self.A_e
        dKdt_de_dClse =  dj_K_e_dClse*self.A_e
        dKdt_de_dCase =  dj_K_e_dCase*self.A_e
        dKdt_de_dVse =  dj_K_e_dVse*self.A_e
        dKdt_de_dNade =  dj_K_mdn_dNade*self.A_m + dj_K_e_dNade*self.A_e
        dKdt_de_dKde =  dj_K_mdn_dKde*self.A_m + dj_K_mdg_dKde*self.A_m + dj_K_e_dKde*self.A_e
        dKdt_de_dClde =  dj_K_mdn_dClde*self.A_m + dj_K_e_dClde*self.A_e
        dKdt_de_dCade =  dj_K_e_dCade*self.A_e
        dKdt_de_dVde =  dj_K_mdn_dVde*self.A_m + dj_K_mdg_dVde*self.A_m + dj_K_e_dVde*self.A_e

        dKdt_de_dq = dj_K_mdn_dq*self.A_m
        dKdt_de_dc = dj_K_mdn_dc*self.A_m
 
        '''dKdt_dg = -j_K_mdg*self.A_m + j_K_ig*self.A_i'''
        dKdt_dg_dNasn = dj_K_ig_dNasn*self.A_i
        dKdt_dg_dKsn = dj_K_ig_dKsn*self.A_i
        dKdt_dg_dClsn = dj_K_ig_dClsn*self.A_i
        dKdt_dg_dCasn = dj_K_ig_dCasn*self.A_i
        dKdt_dg_dVsn =  dj_K_ig_dVsn*self.A_i
        dKdt_dg_dNadn = dj_K_ig_dNadn*self.A_i
        dKdt_dg_dKdn =  dj_K_ig_dKdn*self.A_i
        dKdt_dg_dCldn = dj_K_ig_dCldn*self.A_i
        dKdt_dg_dCadn = dj_K_ig_dCadn*self.A_i
        dKdt_dg_dVdn =  dj_K_ig_dVdn*self.A_i
        dKdt_dg_dNasg = dj_K_ig_dNasg*self.A_i
        dKdt_dg_dKsg = dj_K_ig_dKsg*self.A_i
        dKdt_dg_dClsg = dj_K_ig_dClsg*self.A_i
        dKdt_dg_dVsg = dj_K_ig_dVsg*self.A_i
        dKdt_dg_dNadg = - dj_K_mdg_dNadg*self.A_m + dj_K_ig_dNadg*self.A_i
        dKdt_dg_dKdg = - dj_K_mdg_dKdg*self.A_m + dj_K_ig_dKdg*self.A_i
        dKdt_dg_dCldg =  - dj_K_mdg_dCldg*self.A_m + dj_K_ig_dCldg*self.A_i
        dKdt_dg_dVdg = - dj_K_mdg_dVdg*self.A_m + dj_K_ig_dVdg*self.A_i
        dKdt_dg_dNase = dj_K_ig_dNase*self.A_i
        dKdt_dg_dKse = dj_K_ig_dKse*self.A_i
        dKdt_dg_dClse = dj_K_ig_dClse*self.A_i
        dKdt_dg_dCase = dj_K_ig_dCase*self.A_i
        dKdt_dg_dVse = dj_K_ig_dVse*self.A_i
        dKdt_dg_dNade = dj_K_ig_dNade*self.A_i
        dKdt_dg_dKde =  - dj_K_mdg_dKde*self.A_m + dj_K_ig_dKde*self.A_i
        dKdt_dg_dClde = dj_K_ig_dClde*self.A_i
        dKdt_dg_dCade = dj_K_ig_dCade*self.A_i
        dKdt_dg_dVde = - dj_K_mdg_dVde*self.A_m + dj_K_ig_dVde*self.A_i

        '''dCldt_sn = -j_Cl_msn*self.A_m - j_Cl_in*self.A_i'''
        dCldt_sn_dNasn = -dj_Cl_msn_dNasn*self.A_m - dj_Cl_in_dNasn*self.A_i
        dCldt_sn_dKsn = -dj_Cl_msn_dKsn*self.A_m - dj_Cl_in_dKsn*self.A_i
        dCldt_sn_dClsn = -dj_Cl_msn_dClsn*self.A_m - dj_Cl_in_dClsn*self.A_i
        dCldt_sn_dCasn = -dj_Cl_msn_dCasn*self.A_m - dj_Cl_in_dCasn*self.A_i
        dCldt_sn_dVsn = -dj_Cl_msn_dVsn*self.A_m - dj_Cl_in_dVsn*self.A_i
        dCldt_sn_dNadn = - dj_Cl_in_dNadn*self.A_i
        dCldt_sn_dKdn = - dj_Cl_in_dKdn*self.A_i
        dCldt_sn_dCldn = - dj_Cl_in_dCldn*self.A_i
        dCldt_sn_dCadn = - dj_Cl_in_dCadn*self.A_i
        dCldt_sn_dVdn = - dj_Cl_in_dVdn*self.A_i
        dCldt_sn_dNasg = - dj_Cl_in_dNasg*self.A_i
        dCldt_sn_dKsg = - dj_Cl_in_dKsg*self.A_i
        dCldt_sn_dClsg = - dj_Cl_in_dClsg*self.A_i
        dCldt_sn_dVsg = - dj_Cl_in_dVsg*self.A_i
        dCldt_sn_dNadg = - dj_Cl_in_dNadg*self.A_i
        dCldt_sn_dKdg = - dj_Cl_in_dKdg*self.A_i
        dCldt_sn_dCldg = - dj_Cl_in_dCldg*self.A_i
        dCldt_sn_dVdg = - dj_Cl_in_dVdg*self.A_i
        dCldt_sn_dNase = -dj_Cl_msn_dNase*self.A_m - dj_Cl_in_dNase*self.A_i
        dCldt_sn_dKse = -dj_Cl_msn_dKse*self.A_m - dj_Cl_in_dKse*self.A_i
        dCldt_sn_dClse = -dj_Cl_msn_dClse*self.A_m - dj_Cl_in_dClse*self.A_i
        dCldt_sn_dCase = - dj_Cl_in_dCase*self.A_i
        dCldt_sn_dVse = -dj_Cl_msn_dVse*self.A_m - dj_Cl_in_dVse*self.A_i
        dCldt_sn_dNade = - dj_Cl_in_dNade*self.A_i
        dCldt_sn_dKde = - dj_Cl_in_dKde*self.A_i
        dCldt_sn_dClde = - dj_Cl_in_dClde*self.A_i
        dCldt_sn_dCade = - dj_Cl_in_dCade*self.A_i
        dCldt_sn_dVde = - dj_Cl_in_dVde*self.A_i

        '''dCldt_se = j_Cl_msn*self.A_m + j_Cl_msg*self.A_m - j_Cl_e*self.A_e'''
        dCldt_se_dNasn = dj_Cl_msn_dNasn*self.A_m - dj_Cl_e_dNasn*self.A_e
        dCldt_se_dKsn = dj_Cl_msn_dKsn*self.A_m - dj_Cl_e_dKsn*self.A_e
        dCldt_se_dClsn = dj_Cl_msn_dClsn*self.A_m - dj_Cl_e_dClsn*self.A_e
        dCldt_se_dCasn = dj_Cl_msn_dCasn*self.A_m - dj_Cl_e_dCasn*self.A_e
        dCldt_se_dVsn = dj_Cl_msn_dVsn*self.A_m - dj_Cl_e_dVsn*self.A_e
        dCldt_se_dNadn = - dj_Cl_e_dNadn*self.A_e
        dCldt_se_dKdn = - dj_Cl_e_dKdn*self.A_e
        dCldt_se_dCldn = - dj_Cl_e_dCldn*self.A_e
        dCldt_se_dCadn = - dj_Cl_e_dCadn*self.A_e
        dCldt_se_dVdn = - dj_Cl_e_dVdn*self.A_e
        dCldt_se_dNasg = dj_Cl_msg_dNasg*self.A_m - dj_Cl_e_dNasg*self.A_e
        dCldt_se_dKsg = dj_Cl_msg_dKsg*self.A_m - dj_Cl_e_dKsg*self.A_e
        dCldt_se_dClsg = dj_Cl_msg_dClsg*self.A_m - dj_Cl_e_dClsg*self.A_e
        dCldt_se_dVsg = - dj_Cl_e_dVsg*self.A_e + dj_Cl_msg_dVsg*self.A_m
        dCldt_se_dNadg = - dj_Cl_e_dNadg*self.A_e
        dCldt_se_dKdg = - dj_Cl_e_dKdg*self.A_e
        dCldt_se_dCldg = - dj_Cl_e_dCldg*self.A_e
        dCldt_se_dVdg = - dj_Cl_e_dVdg*self.A_e
        dCldt_se_dNase = dj_Cl_msn_dNase*self.A_m - dj_Cl_e_dNase*self.A_e
        dCldt_se_dKse = dj_Cl_msn_dKse*self.A_m - dj_Cl_e_dKse*self.A_e
        dCldt_se_dClse = dj_Cl_msn_dClse*self.A_m + dj_Cl_msg_dClse*self.A_m - dj_Cl_e_dClse*self.A_e
        dCldt_se_dCase = - dj_Cl_e_dCase*self.A_e
        dCldt_se_dVse = dj_Cl_msn_dVse*self.A_m + dj_Cl_msg_dVse*self.A_m - dj_Cl_e_dVse*self.A_e
        dCldt_se_dNade = - dj_Cl_e_dNade*self.A_e
        dCldt_se_dKde = - dj_Cl_e_dKde*self.A_e
        dCldt_se_dClde = - dj_Cl_e_dClde*self.A_e
        dCldt_se_dCade = - dj_Cl_e_dCade*self.A_e
        dCldt_se_dVde = - dj_Cl_e_dVde*self.A_e

        '''dCldt_sg = -j_Cl_msg*self.A_m - j_Cl_ig*self.A_i'''
        dCldt_sg_dNasn = - dj_Cl_ig_dNasn*self.A_i
        dCldt_sg_dKsn =  - dj_Cl_ig_dKsn*self.A_i
        dCldt_sg_dClsn =  - dj_Cl_ig_dClsn*self.A_i
        dCldt_sg_dCasn = - dj_Cl_ig_dCasn*self.A_i
        dCldt_sg_dVsn =  - dj_Cl_ig_dVsn*self.A_i
        dCldt_sg_dNadn = - dj_Cl_ig_dNadn*self.A_i
        dCldt_sg_dKdn = - dj_Cl_ig_dKdn*self.A_i
        dCldt_sg_dCldn =- dj_Cl_ig_dCldn*self.A_i
        dCldt_sg_dCadn = - dj_Cl_ig_dCadn*self.A_i
        dCldt_sg_dVdn =  - dj_Cl_ig_dVdn*self.A_i
        dCldt_sg_dNasg =  - dj_Cl_msg_dNasg*self.A_m - dj_Cl_ig_dNasg*self.A_i
        dCldt_sg_dKsg = - dj_Cl_msg_dKsg*self.A_m - dj_Cl_ig_dKsg*self.A_i
        dCldt_sg_dClsg =  - dj_Cl_msg_dClsg*self.A_m - dj_Cl_ig_dClsg*self.A_i
        dCldt_sg_dVsg = - dj_Cl_msg_dVsg*self.A_m - dj_Cl_ig_dVsg*self.A_i
        dCldt_sg_dNadg = - dj_Cl_ig_dNadg*self.A_i
        dCldt_sg_dKdg = - dj_Cl_ig_dKdg*self.A_i
        dCldt_sg_dCldg =  - dj_Cl_ig_dCldg*self.A_i
        dCldt_sg_dVdg = - dj_Cl_ig_dVdg*self.A_i
        dCldt_sg_dNase = - dj_Cl_ig_dNase*self.A_i
        dCldt_sg_dKse = - dj_Cl_ig_dKse*self.A_i
        dCldt_sg_dClse = - dj_Cl_msg_dClse*self.A_m - dj_Cl_ig_dClse*self.A_i
        dCldt_sg_dCase = - dj_Cl_ig_dCase*self.A_i
        dCldt_sg_dVse = - dj_Cl_msg_dVse*self.A_m - dj_Cl_ig_dVse*self.A_i
        dCldt_sg_dNade = - dj_Cl_ig_dNade*self.A_i
        dCldt_sg_dKde =  - dj_Cl_ig_dKde*self.A_i
        dCldt_sg_dClde = - dj_Cl_ig_dClde*self.A_i
        dCldt_sg_dCade = - dj_Cl_ig_dCade*self.A_i
        dCldt_sg_dVde = - dj_Cl_ig_dVde*self.A_i

        '''dCldt_dn = -j_Cl_mdn*self.A_m + j_Cl_in*self.A_i'''
        dCldt_dn_dNasn =  dj_Cl_in_dNasn*self.A_i
        dCldt_dn_dKsn = dj_Cl_in_dKsn*self.A_i
        dCldt_dn_dClsn =  dj_Cl_in_dClsn*self.A_i
        dCldt_dn_dCasn = dj_Cl_in_dCasn*self.A_i
        dCldt_dn_dVsn = dj_Cl_in_dVsn*self.A_i
        dCldt_dn_dNadn = -dj_Cl_mdn_dNadn*self.A_m + dj_Cl_in_dNadn*self.A_i
        dCldt_dn_dKdn =  -dj_Cl_mdn_dKdn*self.A_m + dj_Cl_in_dKdn*self.A_i
        dCldt_dn_dCldn = -dj_Cl_mdn_dCldn*self.A_m + dj_Cl_in_dCldn*self.A_i
        dCldt_dn_dCadn = -dj_Cl_mdn_dCadn*self.A_m + dj_Cl_in_dCadn*self.A_i
        dCldt_dn_dVdn = -dj_Cl_mdn_dVdn*self.A_m + dj_Cl_in_dVdn*self.A_i
        dCldt_dn_dNasg = dj_Cl_in_dNasg*self.A_i
        dCldt_dn_dKsg = dj_Cl_in_dKsg*self.A_i
        dCldt_dn_dClsg = dj_Cl_in_dClsg*self.A_i
        dCldt_dn_dVsg = dj_Cl_in_dVsg*self.A_i
        dCldt_dn_dNadg = dj_Cl_in_dNadg*self.A_i
        dCldt_dn_dKdg = dj_Cl_in_dKdg*self.A_i
        dCldt_dn_dCldg = dj_Cl_in_dCldg*self.A_i
        dCldt_dn_dVdg = dj_Cl_in_dVdg*self.A_i
        dCldt_dn_dNase = dj_Cl_in_dNase*self.A_i
        dCldt_dn_dKse = dj_Cl_in_dKse*self.A_i
        dCldt_dn_dClse = dj_Cl_in_dClse*self.A_i
        dCldt_dn_dCase = dj_Cl_in_dCase*self.A_i
        dCldt_dn_dVse = dj_Cl_in_dVse*self.A_i
        dCldt_dn_dNade = -dj_Cl_mdn_dNade*self.A_m + dj_Cl_in_dNade*self.A_i
        dCldt_dn_dKde = -dj_Cl_mdn_dKde*self.A_m + dj_Cl_in_dKde*self.A_i
        dCldt_dn_dClde = -dj_Cl_mdn_dClde*self.A_m + dj_Cl_in_dClde*self.A_i
        dCldt_dn_dCade = dj_Cl_in_dCade*self.A_i
        dCldt_dn_dVde = -dj_Cl_mdn_dVde*self.A_m + dj_Cl_in_dVde*self.A_i

        '''dCldt_de = j_Cl_mdn*self.A_m + j_Cl_mdg*self.A_m + j_Cl_e*self.A_e'''
        dCldt_de_dNasn = dj_Cl_e_dNasn*self.A_e
        dCldt_de_dKsn =  dj_Cl_e_dKsn*self.A_e
        dCldt_de_dClsn =  dj_Cl_e_dClsn*self.A_e
        dCldt_de_dCasn =  dj_Cl_e_dCasn*self.A_e
        dCldt_de_dVsn =  dj_Cl_e_dVsn*self.A_e
        dCldt_de_dNadn =  dj_Cl_mdn_dNadn*self.A_m + dj_Cl_e_dNadn*self.A_e
        dCldt_de_dKdn =  dj_Cl_mdn_dKdn*self.A_m + dj_Cl_e_dKdn*self.A_e
        dCldt_de_dCldn =  dj_Cl_mdn_dCldn*self.A_m + dj_Cl_e_dCldn*self.A_e
        dCldt_de_dCadn =  dj_Cl_mdn_dCadn*self.A_m + dj_Cl_e_dCadn*self.A_e
        dCldt_de_dVdn =  dj_Cl_mdn_dVdn*self.A_m + dj_Cl_e_dVdn*self.A_e
        dCldt_de_dNasg =  dj_Cl_e_dNasg*self.A_e
        dCldt_de_dKsg =  dj_Cl_e_dKsg*self.A_e
        dCldt_de_dClsg =  dj_Cl_e_dClsg*self.A_e
        dCldt_de_dVsg =  dj_Cl_e_dVsg*self.A_e
        dCldt_de_dNadg =  dj_Cl_mdg_dNadg*self.A_m + dj_Cl_e_dNadg*self.A_e
        dCldt_de_dKdg = dj_Cl_mdg_dKdg*self.A_m + dj_Cl_e_dKdg*self.A_e
        dCldt_de_dCldg =  dj_Cl_mdg_dCldg*self.A_m + dj_Cl_e_dCldg*self.A_e
        dCldt_de_dVdg =  dj_Cl_e_dVdg*self.A_e + dj_Cl_mdg_dVdg*self.A_m
        dCldt_de_dNase =  dj_Cl_e_dNase*self.A_e
        dCldt_de_dKse =  dj_Cl_e_dKse*self.A_e
        dCldt_de_dClse =  dj_Cl_e_dClse*self.A_e
        dCldt_de_dCase =  dj_Cl_e_dCase*self.A_e
        dCldt_de_dVse =  dj_Cl_e_dVse*self.A_e
        dCldt_de_dNade =  dj_Cl_mdn_dNade*self.A_m + dj_Cl_e_dNade*self.A_e
        dCldt_de_dKde =  dj_Cl_mdn_dKde*self.A_m + dj_Cl_e_dKde*self.A_e
        dCldt_de_dClde =  dj_Cl_mdn_dClde*self.A_m + dj_Cl_mdg_dClde*self.A_m + dj_Cl_e_dClde*self.A_e
        dCldt_de_dCade =  dj_Cl_e_dCade*self.A_e
        dCldt_de_dVde =  dj_Cl_mdn_dVde*self.A_m + dj_Cl_mdg_dVde*self.A_m+  dj_Cl_e_dVde*self.A_e

        '''dCldt_dg = -j_Cl_mdg*self.A_m + j_Cl_ig*self.A_i'''
        dCldt_dg_dNasn = dj_Cl_ig_dNasn*self.A_i
        dCldt_dg_dKsn = dj_Cl_ig_dKsn*self.A_i
        dCldt_dg_dClsn = dj_Cl_ig_dClsn*self.A_i
        dCldt_dg_dCasn = dj_Cl_ig_dCasn*self.A_i
        dCldt_dg_dVsn =  dj_Cl_ig_dVsn*self.A_i
        dCldt_dg_dNadn = dj_Cl_ig_dNadn*self.A_i
        dCldt_dg_dKdn =  dj_Cl_ig_dKdn*self.A_i
        dCldt_dg_dCldn = dj_Cl_ig_dCldn*self.A_i
        dCldt_dg_dCadn = dj_Cl_ig_dCadn*self.A_i
        dCldt_dg_dVdn =  dj_Cl_ig_dVdn*self.A_i
        dCldt_dg_dNasg = dj_Cl_ig_dNasg*self.A_i
        dCldt_dg_dKsg = dj_Cl_ig_dKsg*self.A_i
        dCldt_dg_dClsg = dj_Cl_ig_dClsg*self.A_i
        dCldt_dg_dVsg = dj_Cl_ig_dVsg*self.A_i
        dCldt_dg_dNadg = - dj_Cl_mdg_dNadg*self.A_m + dj_Cl_ig_dNadg*self.A_i
        dCldt_dg_dKdg = - dj_Cl_mdg_dKdg*self.A_m + dj_Cl_ig_dKdg*self.A_i
        dCldt_dg_dCldg =  - dj_Cl_mdg_dCldg*self.A_m + dj_Cl_ig_dCldg*self.A_i
        dCldt_dg_dVdg = - dj_Cl_mdg_dVdg*self.A_m + dj_Cl_ig_dVdg*self.A_i
        dCldt_dg_dNase = dj_Cl_ig_dNase*self.A_i
        dCldt_dg_dKse = dj_Cl_ig_dKse*self.A_i
        dCldt_dg_dClse = dj_Cl_ig_dClse*self.A_i
        dCldt_dg_dCase = dj_Cl_ig_dCase*self.A_i
        dCldt_dg_dVse = dj_Cl_ig_dVse*self.A_i
        dCldt_dg_dNade = dj_Cl_ig_dNade*self.A_i
        dCldt_dg_dKde =  dj_Cl_ig_dKde*self.A_i
        dCldt_dg_dClde = - dj_Cl_mdg_dClde*self.A_m + dj_Cl_ig_dClde*self.A_i
        dCldt_dg_dCade = dj_Cl_ig_dCade*self.A_i
        dCldt_dg_dVde = - dj_Cl_mdg_dVde*self.A_m + dj_Cl_ig_dVde*self.A_i

        '''dCadt_sn = - j_Ca_in*self.A_i - self.j_Ca_sn()*self.A_m'''
        dCadt_sn_dNasn =  - dj_Ca_in_dNasn*self.A_i
        dCadt_sn_dKsn = - dj_Ca_in_dKsn*self.A_i
        dCadt_sn_dClsn =  - dj_Ca_in_dClsn*self.A_i
        dCadt_sn_dCasn =  - dj_Ca_in_dCasn*self.A_i - dj_Ca_msn_dCasn *self.A_m
        dCadt_sn_dVsn = - dj_Ca_in_dVsn*self.A_i - dj_Ca_msn_dVsn*self.A_m
        dCadt_sn_dNadn =  - dj_Ca_in_dNadn*self.A_i
        dCadt_sn_dKdn = - dj_Ca_in_dKdn*self.A_i
        dCadt_sn_dCldn =  - dj_Ca_in_dCldn*self.A_i
        dCadt_sn_dCadn =  - dj_Ca_in_dCadn*self.A_i
        dCadt_sn_dVdn = - dj_Ca_in_dVdn*self.A_i
        dCadt_sn_dNasg =  - dj_Ca_in_dNasg*self.A_i
        dCadt_sn_dKsg = - dj_Ca_in_dKsg*self.A_i
        dCadt_sn_dClsg =  - dj_Ca_in_dClsg*self.A_i
        dCadt_sn_dVsg = - dj_Ca_in_dVsg*self.A_i
        dCadt_sn_dNadg = - dj_Ca_in_dNadg*self.A_i
        dCadt_sn_dKdg = - dj_Ca_in_dKdg*self.A_i
        dCadt_sn_dCldg =  - dj_Ca_in_dCldg*self.A_i
        dCadt_sn_dVdg = - dj_Ca_in_dVdg*self.A_i
        dCadt_sn_dNase =  - dj_Ca_in_dNase*self.A_i
        dCadt_sn_dKse = - dj_Ca_in_dKse*self.A_i
        dCadt_sn_dClse =  - dj_Ca_in_dClse*self.A_i
        dCadt_sn_dCase =  - dj_Ca_in_dCase*self.A_i
        dCadt_sn_dVse = - dj_Ca_in_dVse*self.A_i
        dCadt_sn_dNade =  - dj_Ca_in_dNade*self.A_i
        dCadt_sn_dKde = - dj_Ca_in_dKde*self.A_i
        dCadt_sn_dClde =  - dj_Ca_in_dClde*self.A_i
        dCadt_sn_dCade =  - dj_Ca_in_dCade*self.A_i
        dCadt_sn_dVde = - dj_Ca_in_dVde*self.A_i

        '''dCadt_se = - j_Ca_e*self.A_e + self.j_Ca_sn()*self.A_m'''
        dCadt_se_dNasn =  - dj_Ca_e_dNasn*self.A_e
        dCadt_se_dKsn = - dj_Ca_e_dKsn*self.A_e
        dCadt_se_dClsn =  - dj_Ca_e_dClsn*self.A_e
        dCadt_se_dCasn =  - dj_Ca_e_dCasn*self.A_e + dj_Ca_msn_dCasn *self.A_m
        dCadt_se_dVsn = - dj_Ca_e_dVsn*self.A_e + dj_Ca_msn_dVsn*self.A_m
        dCadt_se_dNadn =  - dj_Ca_e_dNadn*self.A_e
        dCadt_se_dKdn = - dj_Ca_e_dKdn*self.A_e
        dCadt_se_dCldn =  - dj_Ca_e_dCldn*self.A_e
        dCadt_se_dCadn =  - dj_Ca_e_dCadn*self.A_e
        dCadt_se_dVdn = - dj_Ca_e_dVdn*self.A_e
        dCadt_se_dNasg =  - dj_Ca_e_dNasg*self.A_e
        dCadt_se_dKsg = - dj_Ca_e_dKsg*self.A_e
        dCadt_se_dClsg =  - dj_Ca_e_dClsg*self.A_e
        dCadt_se_dVsg = - dj_Ca_e_dVsg*self.A_e
        dCadt_se_dNadg = - dj_Ca_e_dNadg*self.A_e
        dCadt_se_dKdg = - dj_Ca_e_dKdg*self.A_e
        dCadt_se_dCldg =  - dj_Ca_e_dCldg*self.A_e
        dCadt_se_dVdg = - dj_Ca_e_dVdg*self.A_e
        dCadt_se_dNase =  - dj_Ca_e_dNase*self.A_e
        dCadt_se_dKse = - dj_Ca_e_dKse*self.A_e
        dCadt_se_dClse =  - dj_Ca_e_dClse*self.A_e
        dCadt_se_dCase =  - dj_Ca_e_dCase*self.A_e
        dCadt_se_dVse = - dj_Ca_e_dVse*self.A_e
        dCadt_se_dNade =  - dj_Ca_e_dNade*self.A_e
        dCadt_se_dKde = - dj_Ca_e_dKde*self.A_e
        dCadt_se_dClde =  - dj_Ca_e_dClde*self.A_e
        dCadt_se_dCade =  - dj_Ca_e_dCade*self.A_e
        dCadt_se_dVde = - dj_Ca_e_dVde*self.A_e

        '''dCadt_dn = j_Ca_in*self.A_i - j_Ca_mdn*self.A_m '''
        dCadt_dn_dNasn =  dj_Ca_in_dNasn*self.A_i
        dCadt_dn_dKsn = dj_Ca_in_dKsn*self.A_i
        dCadt_dn_dClsn =  dj_Ca_in_dClsn*self.A_i
        dCadt_dn_dCasn = dj_Ca_in_dCasn*self.A_i
        dCadt_dn_dVsn = dj_Ca_in_dVsn*self.A_i
        dCadt_dn_dNadn = -dj_Ca_mdn_dNadn*self.A_m + dj_Ca_in_dNadn*self.A_i
        dCadt_dn_dKdn =  -dj_Ca_mdn_dKdn*self.A_m + dj_Ca_in_dKdn*self.A_i
        dCadt_dn_dCldn = -dj_Ca_mdn_dCldn*self.A_m + dj_Ca_in_dCldn*self.A_i
        dCadt_dn_dCadn = -dj_Ca_mdn_dCadn*self.A_m + dj_Ca_in_dCadn*self.A_i
        dCadt_dn_dVdn = -dj_Ca_mdn_dVdn*self.A_m + dj_Ca_in_dVdn*self.A_i
        dCadt_dn_dNasg = dj_Ca_in_dNasg*self.A_i
        dCadt_dn_dKsg = dj_Ca_in_dKsg*self.A_i
        dCadt_dn_dClsg = dj_Ca_in_dClsg*self.A_i
        dCadt_dn_dVsg = dj_Ca_in_dVsg*self.A_i
        dCadt_dn_dNadg = dj_Ca_in_dNadg*self.A_i
        dCadt_dn_dKdg = dj_Ca_in_dKdg*self.A_i
        dCadt_dn_dCldg = dj_Ca_in_dCldg*self.A_i
        dCadt_dn_dVdg = dj_Ca_in_dVdg*self.A_i
        dCadt_dn_dNase = dj_Ca_in_dNase*self.A_i
        dCadt_dn_dKse = dj_Ca_in_dKse*self.A_i
        dCadt_dn_dClse = dj_Ca_in_dClse*self.A_i
        dCadt_dn_dCase = dj_Ca_in_dCase*self.A_i
        dCadt_dn_dVse = dj_Ca_in_dVse*self.A_i
        dCadt_dn_dNade = dj_Ca_in_dNade*self.A_i
        dCadt_dn_dKde = dj_Ca_in_dKde*self.A_i
        dCadt_dn_dClde = dj_Ca_in_dClde*self.A_i
        dCadt_dn_dCade = -dj_Ca_mdn_dCade*self.A_m + dj_Ca_in_dCade*self.A_i
        dCadt_dn_dVde = -dj_Ca_mdn_dVde*self.A_m +  dj_Ca_in_dVde*self.A_i  

        dCadt_dn_ds = -dj_Ca_mdn_ds*self.A_m
        dCadt_dn_dz =  -dj_Ca_mdn_dz*self.A_m 

        '''dCadt_de = j_Ca_e*self.A_e + j_Ca_mdn*self.A_m '''
        dCadt_de_dNasn =  dj_Ca_e_dNasn*self.A_e
        dCadt_de_dKsn = dj_Ca_e_dKsn*self.A_e
        dCadt_de_dClsn =  dj_Ca_e_dClsn*self.A_e
        dCadt_de_dCasn =  dj_Ca_e_dCasn*self.A_e
        dCadt_de_dVsn = dj_Ca_e_dVsn*self.A_e
        dCadt_de_dNadn =  dj_Ca_e_dNadn*self.A_e + dj_Ca_mdn_dNadn*self.A_m
        dCadt_de_dKdn = dj_Ca_e_dKdn*self.A_e + dj_Ca_mdn_dKdn*self.A_m
        dCadt_de_dCldn =  dj_Ca_e_dCldn*self.A_e + dj_Ca_mdn_dCldn*self.A_m
        dCadt_de_dCadn =  dj_Ca_e_dCadn*self.A_e + dj_Ca_mdn_dCadn*self.A_m
        dCadt_de_dVdn = dj_Ca_e_dVdn*self.A_e + dj_Ca_mdn_dVdn*self.A_m
        dCadt_de_dNasg =  dj_Ca_e_dNasg*self.A_e
        dCadt_de_dKsg = dj_Ca_e_dKsg*self.A_e
        dCadt_de_dClsg =  dj_Ca_e_dClsg*self.A_e
        dCadt_de_dVsg = dj_Ca_e_dVsg*self.A_e
        dCadt_de_dNadg = dj_Ca_e_dNadg*self.A_e
        dCadt_de_dKdg = dj_Ca_e_dKdg*self.A_e
        dCadt_de_dCldg =  dj_Ca_e_dCldg*self.A_e
        dCadt_de_dVdg = dj_Ca_e_dVdg*self.A_e
        dCadt_de_dNase =  dj_Ca_e_dNase*self.A_e
        dCadt_de_dKse = dj_Ca_e_dKse*self.A_e
        dCadt_de_dClse =  dj_Ca_e_dClse*self.A_e
        dCadt_de_dCase =  dj_Ca_e_dCase*self.A_e
        dCadt_de_dVse = dj_Ca_e_dVse*self.A_e
        dCadt_de_dNade =  dj_Ca_e_dNade*self.A_e
        dCadt_de_dKde = dj_Ca_e_dKde*self.A_e
        dCadt_de_dClde =  dj_Ca_e_dClde*self.A_e
        dCadt_de_dCade =  dj_Ca_e_dCade*self.A_e + dj_Ca_mdn_dCade*self.A_m
        dCadt_de_dVde = dj_Ca_e_dVde*self.A_e + dj_Ca_mdn_dVde*self.A_m

        dCadt_de_ds = dj_Ca_mdn_ds*self.A_m
        dCadt_de_dz =  dj_Ca_mdn_dz*self.A_m 

        # Gating variables derivatives 

        '''dndt = self.alpha_n(phi_msn)*(1.0-self.n) - self.beta_n(phi_msn)*self.n'''
        dndt_dNasn =  self.dalpha_n(phi_msn,dphi_msn_dNasn)*(1.0-self.n) - self.dbeta_n(phi_msn,dphi_msn_dNasn)*self.n
        dndt_dKsn = self.dalpha_n(phi_msn,dphi_msn_dKsn)*(1.0-self.n) - self.dbeta_n(phi_msn,dphi_msn_dKsn)*self.n
        dndt_dClsn =  self.dalpha_n(phi_msn,dphi_msn_dClsn)*(1.0-self.n) - self.dbeta_n(phi_msn,dphi_msn_dClsn)*self.n
        dndt_dCasn =  self.dalpha_n(phi_msn,dphi_msn_dCasn)*(1.0-self.n) - self.dbeta_n(phi_msn,dphi_msn_dCasn)*self.n

        dndt_dn = - self.alpha_n(phi_msn) - self.beta_n(phi_msn)

        '''dhdt = self.alpha_h(phi_msn)*(1.0-self.h) - self.beta_h(phi_msn)*self.h '''
        dhdt_dNasn =  self.dalpha_h(phi_msn,dphi_msn_dNasn)*(1.0-self.h) - self.dbeta_h(phi_msn,dphi_msn_dNasn)*self.h
        dhdt_dKsn = self.dalpha_h(phi_msn,dphi_msn_dKsn)*(1.0-self.h) - self.dbeta_h(phi_msn,dphi_msn_dKsn)*self.h
        dhdt_dClsn =  self.dalpha_h(phi_msn,dphi_msn_dClsn)*(1.0-self.h) - self.dbeta_h(phi_msn,dphi_msn_dClsn)*self.h
        dhdt_dCasn =  self.dalpha_h(phi_msn,dphi_msn_dCasn)*(1.0-self.h) - self.dbeta_h(phi_msn,dphi_msn_dCasn)*self.h

        dhdt_dh = - self.alpha_h(phi_msn) - self.beta_h(phi_msn)

        '''dsdt = self.alpha_s(phi_mdn)*(1.0-self.s) - self.beta_s(phi_mdn)*self.s'''
        dsdt_dNadn =  self.dalpha_s(phi_mdn,dphi_mdn_dNadn)*(1.0-self.s) - self.dbeta_s(phi_mdn,dphi_mdn_dNadn)*self.s
        dsdt_dKdn = self.dalpha_s(phi_mdn,dphi_mdn_dKdn)*(1.0-self.s) - self.dbeta_s(phi_mdn,dphi_mdn_dKdn)*self.s
        dsdt_dCldn =  self.dalpha_s(phi_mdn,dphi_mdn_dCldn)*(1.0-self.s) - self.dbeta_s(phi_mdn,dphi_mdn_dCldn)*self.s
        dsdt_dCadn =  self.dalpha_s(phi_mdn,dphi_mdn_dCadn)*(1.0-self.s) - self.dbeta_s(phi_mdn,dphi_mdn_dCadn)*self.s

        dsdt_ds = - self.alpha_s(phi_mdn) - self.beta_s(phi_mdn)

        '''dcdt = self.alpha_c(phi_mdn)*(1.0-self.c) - self.beta_c(phi_mdn)*self.c'''
        dcdt_dNadn =  self.dalpha_c(phi_mdn,dphi_mdn_dNadn)*(1.0-self.c) - self.dbeta_c(phi_mdn,dphi_mdn_dNadn)*self.c
        dcdt_dKdn = self.dalpha_c(phi_mdn,dphi_mdn_dKdn)*(1.0-self.c) - self.dbeta_c(phi_mdn,dphi_mdn_dKdn)*self.c
        dcdt_dCldn =  self.dalpha_c(phi_mdn,dphi_mdn_dCldn)*(1.0-self.c) - self.dbeta_c(phi_mdn,dphi_mdn_dCldn)*self.c
        dcdt_dCadn =  self.dalpha_c(phi_mdn,dphi_mdn_dCadn)*(1.0-self.c) - self.dbeta_c(phi_mdn,dphi_mdn_dCadn)*self.c

        dcdt_dc = - self.alpha_c(phi_mdn) - self.beta_c(phi_mdn)

        '''dqdt = self.alpha_q()*(1.0-self.q) - self.beta_q()*self.q'''
        dqdt_dCadn = self.dalpha_q_dCa_dn()*(1.0-self.q)
        dqdt_dVdn = self.dalpha_q_dV_dn()*(1.0-self.q)

        dqdt_dq = - self.alpha_q() -self.beta_q()

        '''dzdt = (self.z_inf(phi_mdn) - self.z)/1e3'''
        dzdt_dNadn = self.dz_inf(phi_mdn, dphi_mdn_dNadn)/1e3
        dzdt_dKdn = self.dz_inf(phi_mdn, dphi_mdn_dKdn)/1e3
        dzdt_dCldn = self.dz_inf(phi_mdn, dphi_mdn_dCldn)/1e3
        dzdt_dCadn = self.dz_inf(phi_mdn, dphi_mdn_dCadn)/1e3

        dzdt_dz = -1.0/1e3

        # Volumes derivatives 
        
        '''dVsidt = self.G_n * (self.psi_se - self.psi_sn)'''
        dVsidt_dNasn = - self.G_n * self.dpsi_dck(dcNa_sn_dNasn)
        dVsidt_dKsn = - self.G_n * self.dpsi_dck(dcK_sn_dKsn)
        dVsidt_dClsn = - self.G_n * self.dpsi_dck(dcCl_sn_dClsn)
        dVsidt_dCasn = - self.G_n * self.dpsi_dck(dcCa_sn_dCasn)
        dVsidt_dVsn = - self.G_n * (self.dpsi_dck(dcNa_sn_dVsn) + self.dpsi_dck(dcK_sn_dVsn) + self.dpsi_dck(dcCl_sn_dVsn) + self.dpsi_dck(dcCa_sn_dVsn))
        dVsidt_dNase = self.G_n * self.dpsi_dck(dcNa_se_dNase)
        dVsidt_dKse = self.G_n * self.dpsi_dck(dcK_se_dKse)
        dVsidt_dClse = self.G_n * self.dpsi_dck(dcCl_se_dClse)
        dVsidt_dCase = self.G_n * self.dpsi_dck(dcCa_se_dCase)
        dVsidt_dVse = self.G_n * (self.dpsi_dck(dcNa_se_dVse) + self.dpsi_dck(dcK_se_dVse) + self.dpsi_dck(dcCl_se_dVse) + self.dpsi_dck(dcCa_se_dVse))

        '''dVsgdt = self.G_g * (self.psi_se - self.psi_sg)'''
        dVsgdt_dNasg = - self.G_g * self.dpsi_dck(dcNa_sg_dNasg)
        dVsgdt_dKsg = - self.G_g * self.dpsi_dck(dcK_sg_dKsg)
        dVsgdt_dClsg = - self.G_g * self.dpsi_dck(dcCl_sg_dClsg)
        dVsgdt_dVsg = - self.G_g * (self.dpsi_dck(dcNa_sg_dVsg) + self.dpsi_dck(dcK_sg_dVsg) + self.dpsi_dck(dcCl_sg_dVsg))
        dVsgdt_dNase = self.G_g * self.dpsi_dck(dcNa_se_dNase)
        dVsgdt_dKse = self.G_g * self.dpsi_dck(dcK_se_dKse)
        dVsgdt_dClse = self.G_g * self.dpsi_dck(dcCl_se_dClse)
        dVsgdt_dCase = self.G_g * self.dpsi_dck(dcCa_se_dCase)
        dVsgdt_dVse = self.G_g * (self.dpsi_dck(dcNa_se_dVse) + self.dpsi_dck(dcK_se_dVse) + self.dpsi_dck(dcCl_se_dVse) + self.dpsi_dck(dcCa_se_dVse))

        '''dVdidt = self.G_n * (self.psi_de - self.psi_dn)'''
        dVdidt_dNadn = - self.G_n * self.dpsi_dck(dcNa_dn_dNadn)
        dVdidt_dKdn = - self.G_n * self.dpsi_dck(dcK_dn_dKdn)
        dVdidt_dCldn = - self.G_n * self.dpsi_dck(dcCl_dn_dCldn)
        dVdidt_dCadn = - self.G_n * self.dpsi_dck(dcCa_dn_dCadn)
        dVdidt_dVdn = - self.G_n * (self.dpsi_dck(dcNa_dn_dVdn) + self.dpsi_dck(dcK_dn_dVdn) + self.dpsi_dck(dcCl_dn_dVdn) + self.dpsi_dck(dcCa_dn_dVdn))
        dVdidt_dNade = self.G_n * self.dpsi_dck(dcNa_de_dNade)
        dVdidt_dKde = self.G_n * self.dpsi_dck(dcK_de_dKde)
        dVdidt_dClde = self.G_n * self.dpsi_dck(dcCl_de_dClde)
        dVdidt_dCade = self.G_n * self.dpsi_dck(dcCa_de_dCade)
        dVdidt_dVde = self.G_n * (self.dpsi_dck(dcNa_de_dVde) + self.dpsi_dck(dcK_de_dVde) + self.dpsi_dck(dcCl_de_dVde) + self.dpsi_dck(dcCa_de_dVde))
        
        '''dVdgdt = self.G_g * (self.psi_de - self.psi_dg)'''
        dVdgdt_dNadg = - self.G_g * self.dpsi_dck(dcNa_dg_dNadg)
        dVdgdt_dKdg = - self.G_g * self.dpsi_dck(dcK_dg_dKdg)
        dVdgdt_dCldg = - self.G_g * self.dpsi_dck(dcCl_dg_dCldg)
        dVdgdt_dVdg = - self.G_g * (self.dpsi_dck(dcNa_dg_dVdg) + self.dpsi_dck(dcK_dg_dVdg) + self.dpsi_dck(dcCl_dg_dVdg))
        dVdgdt_dNade = self.G_g * self.dpsi_dck(dcNa_de_dNade)
        dVdgdt_dKde = self.G_g * self.dpsi_dck(dcK_de_dKde)
        dVdgdt_dClde = self.G_g * self.dpsi_dck(dcCl_de_dClde)
        dVdgdt_dCade = self.G_g * self.dpsi_dck(dcCa_de_dCade)
        dVdgdt_dVde = self.G_g * (self.dpsi_dck(dcNa_de_dVde) + self.dpsi_dck(dcK_de_dVde) + self.dpsi_dck(dcCl_de_dVde) + self.dpsi_dck(dcCa_de_dVde))

        '''dVsedt = - (dVsidt + dVsgdt)'''
        dVsedt_dNasn = - dVsidt_dNasn
        dVsedt_dKsn = - dVsidt_dKsn
        dVsedt_dClsn = - dVsidt_dClsn
        dVsedt_dCasn = - dVsidt_dCasn
        dVsedt_dVsn = - dVsidt_dVsn
        dVsedt_dNase = - (dVsidt_dNase + dVsgdt_dNase)
        dVsedt_dKse = - (dVsidt_dKse + dVsgdt_dKse)
        dVsedt_dClse = - (dVsidt_dClse + dVsgdt_dClse)
        dVsedt_dCase = - (dVsidt_dCase + dVsgdt_dCase)
        dVsedt_dVse = - (dVsidt_dVse + dVsgdt_dVse)
        dVsedt_dNasg = - dVsgdt_dNasg
        dVsedt_dKsg = - dVsgdt_dKsg
        dVsedt_dClsg = - dVsgdt_dClsg
        dVsedt_dVsg = - dVsgdt_dVsg

        '''dVdedt = - (dVdidt + dVdgdt)'''
        dVdedt_dNadn = - dVdidt_dNadn
        dVdedt_dKdn = - dVdidt_dKdn
        dVdedt_dCldn = - dVdidt_dCldn
        dVdedt_dCadn = - dVdidt_dCadn
        dVdedt_dVdn = - dVdidt_dVdn
        dVdedt_dNade = - (dVdidt_dNade + dVdgdt_dNade)
        dVdedt_dKde = - (dVdidt_dKde + dVdgdt_dKde)
        dVdedt_dClde = - (dVdidt_dClde + dVdgdt_dClde)
        dVdedt_dCade = - (dVdidt_dCade + dVdgdt_dCade)
        dVdedt_dVde = - (dVdidt_dVde + dVdgdt_dVde)
        dVdedt_dNadg = - dVdgdt_dNadg
        dVdedt_dKdg = - dVdgdt_dKdg
        dVdedt_dCldg = - dVdgdt_dCldg
        dVdedt_dVdg = - dVdgdt_dVdg

        # Build dense Jacobian matrix
        if dense:
            dNadt_sn_dx = [dNadt_sn_dNasn , dNadt_sn_dNase , dNadt_sn_dNasg , dNadt_sn_dNadn , dNadt_sn_dNade , dNadt_sn_dNadg , \
                            dNadt_sn_dKsn , dNadt_sn_dKse , dNadt_sn_dKsg , dNadt_sn_dKdn , dNadt_sn_dKde , dNadt_sn_dKdg , \
                            dNadt_sn_dClsn , dNadt_sn_dClse , dNadt_sn_dClsg , dNadt_sn_dCldn , dNadt_sn_dClde , dNadt_sn_dCldg , \
                            dNadt_sn_dCasn , dNadt_sn_dCase , dNadt_sn_dCadn , dNadt_sn_dCade , \
                            0 , dNadt_sn_dh , 0 , 0 , 0 , 0 , \
                            dNadt_sn_dVsn , dNadt_sn_dVse , dNadt_sn_dVsg , dNadt_sn_dVdn , dNadt_sn_dVde , dNadt_sn_dVdg]
            dNadt_se_dx = [dNadt_se_dNasn , dNadt_se_dNase , dNadt_se_dNasg , dNadt_se_dNadn , dNadt_se_dNade , dNadt_se_dNadg , \
                                    dNadt_se_dKsn , dNadt_se_dKse , dNadt_se_dKsg , dNadt_se_dKdn , dNadt_se_dKde , dNadt_se_dKdg , \
                                    dNadt_se_dClsn , dNadt_se_dClse , dNadt_se_dClsg , dNadt_se_dCldn , dNadt_se_dClde , dNadt_se_dCldg , \
                                    dNadt_se_dCasn , dNadt_se_dCase , dNadt_se_dCadn , dNadt_se_dCade , \
                                    0 , dNadt_se_dh , 0 , 0 , 0 , 0 , \
                                    dNadt_se_dVsn , dNadt_se_dVse , dNadt_se_dVsg , dNadt_se_dVdn , dNadt_se_dVde , dNadt_se_dVdg]
            dNadt_sg_dx = [dNadt_sg_dNasn , dNadt_sg_dNase , dNadt_sg_dNasg , dNadt_sg_dNadn , dNadt_sg_dNade , dNadt_sg_dNadg , \
                                    dNadt_sg_dKsn , dNadt_sg_dKse , dNadt_sg_dKsg , dNadt_sg_dKdn , dNadt_sg_dKde , dNadt_sg_dKdg , \
                                    dNadt_sg_dClsn , dNadt_sg_dClse , dNadt_sg_dClsg , dNadt_sg_dCldn , dNadt_sg_dClde , dNadt_sg_dCldg , \
                                    dNadt_sg_dCasn , dNadt_sg_dCase , dNadt_sg_dCadn , dNadt_sg_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dNadt_sg_dVsn , dNadt_sg_dVse , dNadt_sg_dVsg , dNadt_sg_dVdn , dNadt_sg_dVde , dNadt_sg_dVdg]
            dNadt_dn_dx = [dNadt_dn_dNasn , dNadt_dn_dNase , dNadt_dn_dNasg , dNadt_dn_dNadn , dNadt_dn_dNade , dNadt_dn_dNadg , \
                                    dNadt_dn_dKsn , dNadt_dn_dKse , dNadt_dn_dKsg , dNadt_dn_dKdn , dNadt_dn_dKde , dNadt_dn_dKdg , \
                                    dNadt_dn_dClsn , dNadt_dn_dClse , dNadt_dn_dClsg , dNadt_dn_dCldn , dNadt_dn_dClde , dNadt_dn_dCldg , \
                                    dNadt_dn_dCasn , dNadt_dn_dCase , dNadt_dn_dCadn , dNadt_dn_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dNadt_dn_dVsn , dNadt_dn_dVse , dNadt_dn_dVsg , dNadt_dn_dVdn , dNadt_dn_dVde , dNadt_dn_dVdg]
            dNadt_de_dx = [dNadt_de_dNasn , dNadt_de_dNase , dNadt_de_dNasg , dNadt_de_dNadn , dNadt_de_dNade , dNadt_de_dNadg , \
                                    dNadt_de_dKsn , dNadt_de_dKse , dNadt_de_dKsg , dNadt_de_dKdn , dNadt_de_dKde , dNadt_de_dKdg , \
                                    dNadt_de_dClsn , dNadt_de_dClse , dNadt_de_dClsg , dNadt_de_dCldn , dNadt_de_dClde , dNadt_de_dCldg , \
                                    dNadt_de_dCasn , dNadt_de_dCase , dNadt_de_dCadn , dNadt_de_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dNadt_de_dVsn , dNadt_de_dVse , dNadt_de_dVsg , dNadt_de_dVdn , dNadt_de_dVde , dNadt_de_dVdg]
            dNadt_dg_dx = [dNadt_dg_dNasn , dNadt_dg_dNase , dNadt_dg_dNasg , dNadt_dg_dNadn , dNadt_dg_dNade , dNadt_dg_dNadg , \
                                    dNadt_dg_dKsn , dNadt_dg_dKse , dNadt_dg_dKsg , dNadt_dg_dKdn , dNadt_dg_dKde , dNadt_dg_dKdg , \
                                    dNadt_dg_dClsn , dNadt_dg_dClse , dNadt_dg_dClsg , dNadt_dg_dCldn , dNadt_dg_dClde , dNadt_dg_dCldg , \
                                    dNadt_dg_dCasn , dNadt_dg_dCase , dNadt_dg_dCadn , dNadt_dg_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dNadt_dg_dVsn , dNadt_dg_dVse , dNadt_dg_dVsg , dNadt_dg_dVdn , dNadt_dg_dVde , dNadt_dg_dVdg]
            dKdt_sn_dx = [dKdt_sn_dNasn , dKdt_sn_dNase , dKdt_sn_dNasg , dKdt_sn_dNadn , dKdt_sn_dNade , dKdt_sn_dNadg , \
                                    dKdt_sn_dKsn , dKdt_sn_dKse , dKdt_sn_dKsg , dKdt_sn_dKdn , dKdt_sn_dKde , dKdt_sn_dKdg , \
                                    dKdt_sn_dClsn , dKdt_sn_dClse , dKdt_sn_dClsg , dKdt_sn_dCldn , dKdt_sn_dClde , dKdt_sn_dCldg , \
                                    dKdt_sn_dCasn , dKdt_sn_dCase , dKdt_sn_dCadn , dKdt_sn_dCade , \
                                    dKdt_sn_dn , 0 , 0 , 0 , 0 , 0 , \
                                    dKdt_sn_dVsn , dKdt_sn_dVse , dKdt_sn_dVsg , dKdt_sn_dVdn , dKdt_sn_dVde , dKdt_sn_dVdg]
            dKdt_se_dx = [dKdt_se_dNasn , dKdt_se_dNase , dKdt_se_dNasg , dKdt_se_dNadn , dKdt_se_dNade , dKdt_se_dNadg , \
                                    dKdt_se_dKsn , dKdt_se_dKse , dKdt_se_dKsg , dKdt_se_dKdn , dKdt_se_dKde , dKdt_se_dKdg , \
                                    dKdt_se_dClsn , dKdt_se_dClse , dKdt_se_dClsg , dKdt_se_dCldn , dKdt_se_dClde , dKdt_se_dCldg , \
                                    dKdt_se_dCasn , dKdt_se_dCase , dKdt_se_dCadn , dKdt_se_dCade , \
                                    dKdt_se_dn , 0 , 0 , 0 , 0 , 0 , \
                                    dKdt_se_dVsn , dKdt_se_dVse , dKdt_se_dVsg , dKdt_se_dVdn , dKdt_se_dVde , dKdt_se_dVdg]
            dKdt_sg_dx = [dKdt_sg_dNasn , dKdt_sg_dNase , dKdt_sg_dNasg , dKdt_sg_dNadn , dKdt_sg_dNade , dKdt_sg_dNadg , \
                                    dKdt_sg_dKsn , dKdt_sg_dKse , dKdt_sg_dKsg , dKdt_sg_dKdn , dKdt_sg_dKde , dKdt_sg_dKdg , \
                                    dKdt_sg_dClsn , dKdt_sg_dClse , dKdt_sg_dClsg , dKdt_sg_dCldn , dKdt_sg_dClde , dKdt_sg_dCldg , \
                                    dKdt_sg_dCasn , dKdt_sg_dCase , dKdt_sg_dCadn , dKdt_sg_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dKdt_sg_dVsn , dKdt_sg_dVse , dKdt_sg_dVsg , dKdt_sg_dVdn , dKdt_sg_dVde , dKdt_sg_dVdg]
            dKdt_dn_dx = [dKdt_dn_dNasn , dKdt_dn_dNase , dKdt_dn_dNasg , dKdt_dn_dNadn , dKdt_dn_dNade , dKdt_dn_dNadg , \
                                    dKdt_dn_dKsn , dKdt_dn_dKse , dKdt_dn_dKsg , dKdt_dn_dKdn , dKdt_dn_dKde , dKdt_dn_dKdg , \
                                    dKdt_dn_dClsn , dKdt_dn_dClse , dKdt_dn_dClsg , dKdt_dn_dCldn , dKdt_dn_dClde , dKdt_dn_dCldg , \
                                    dKdt_dn_dCasn , dKdt_dn_dCase , dKdt_dn_dCadn , dKdt_dn_dCade , \
                                    0 , 0 , 0 , dKdt_dn_dc , dKdt_dn_dq , 0 , \
                                    dKdt_dn_dVsn , dKdt_dn_dVse , dKdt_dn_dVsg , dKdt_dn_dVdn , dKdt_dn_dVde , dKdt_dn_dVdg]
            dKdt_de_dx = [dKdt_de_dNasn , dKdt_de_dNase , dKdt_de_dNasg , dKdt_de_dNadn , dKdt_de_dNade , dKdt_de_dNadg , \
                                    dKdt_de_dKsn , dKdt_de_dKse , dKdt_de_dKsg , dKdt_de_dKdn , dKdt_de_dKde , dKdt_de_dKdg , \
                                    dKdt_de_dClsn , dKdt_de_dClse , dKdt_de_dClsg , dKdt_de_dCldn , dKdt_de_dClde , dKdt_de_dCldg , \
                                    dKdt_de_dCasn , dKdt_de_dCase , dKdt_de_dCadn , dKdt_de_dCade , \
                                    0 , 0 , 0 , dKdt_de_dc , dKdt_de_dq , 0 , \
                                    dKdt_de_dVsn , dKdt_de_dVse , dKdt_de_dVsg , dKdt_de_dVdn , dKdt_de_dVde , dKdt_de_dVdg]
            dKdt_dg_dx = [dKdt_dg_dNasn , dKdt_dg_dNase , dKdt_dg_dNasg , dKdt_dg_dNadn , dKdt_dg_dNade , dKdt_dg_dNadg , \
                                    dKdt_dg_dKsn , dKdt_dg_dKse , dKdt_dg_dKsg , dKdt_dg_dKdn , dKdt_dg_dKde , dKdt_dg_dKdg , \
                                    dKdt_dg_dClsn , dKdt_dg_dClse , dKdt_dg_dClsg , dKdt_dg_dCldn , dKdt_dg_dClde , dKdt_dg_dCldg , \
                                    dKdt_dg_dCasn , dKdt_dg_dCase , dKdt_dg_dCadn , dKdt_dg_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dKdt_dg_dVsn , dKdt_dg_dVse , dKdt_dg_dVsg , dKdt_dg_dVdn , dKdt_dg_dVde , dKdt_dg_dVdg]
            dCldt_sn_dx = [dCldt_sn_dNasn , dCldt_sn_dNase , dCldt_sn_dNasg , dCldt_sn_dNadn , dCldt_sn_dNade , dCldt_sn_dNadg , \
                                    dCldt_sn_dKsn , dCldt_sn_dKse , dCldt_sn_dKsg , dCldt_sn_dKdn , dCldt_sn_dKde , dCldt_sn_dKdg , \
                                    dCldt_sn_dClsn , dCldt_sn_dClse , dCldt_sn_dClsg , dCldt_sn_dCldn , dCldt_sn_dClde , dCldt_sn_dCldg , \
                                    dCldt_sn_dCasn , dCldt_sn_dCase , dCldt_sn_dCadn , dCldt_sn_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dCldt_sn_dVsn , dCldt_sn_dVse , dCldt_sn_dVsg , dCldt_sn_dVdn , dCldt_sn_dVde , dCldt_sn_dVdg]
            dCldt_se_dx = [dCldt_se_dNasn , dCldt_se_dNase , dCldt_se_dNasg , dCldt_se_dNadn , dCldt_se_dNade , dCldt_se_dNadg , \
                                    dCldt_se_dKsn , dCldt_se_dKse , dCldt_se_dKsg , dCldt_se_dKdn , dCldt_se_dKde , dCldt_se_dKdg , \
                                    dCldt_se_dClsn , dCldt_se_dClse , dCldt_se_dClsg , dCldt_se_dCldn , dCldt_se_dClde , dCldt_se_dCldg , \
                                    dCldt_se_dCasn , dCldt_se_dCase , dCldt_se_dCadn , dCldt_se_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dCldt_se_dVsn , dCldt_se_dVse , dCldt_se_dVsg , dCldt_se_dVdn , dCldt_se_dVde , dCldt_se_dVdg]
            dCldt_sg_dx = [dCldt_sg_dNasn , dCldt_sg_dNase , dCldt_sg_dNasg , dCldt_sg_dNadn , dCldt_sg_dNade , dCldt_sg_dNadg , \
                                    dCldt_sg_dKsn , dCldt_sg_dKse , dCldt_sg_dKsg , dCldt_sg_dKdn , dCldt_sg_dKde , dCldt_sg_dKdg , \
                                    dCldt_sg_dClsn , dCldt_sg_dClse , dCldt_sg_dClsg , dCldt_sg_dCldn , dCldt_sg_dClde , dCldt_sg_dCldg , \
                                    dCldt_sg_dCasn , dCldt_sg_dCase , dCldt_sg_dCadn , dCldt_sg_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dCldt_sg_dVsn , dCldt_sg_dVse , dCldt_sg_dVsg , dCldt_sg_dVdn , dCldt_sg_dVde , dCldt_sg_dVdg]
            dCldt_dn_dx = [dCldt_dn_dNasn , dCldt_dn_dNase , dCldt_dn_dNasg , dCldt_dn_dNadn , dCldt_dn_dNade , dCldt_dn_dNadg , \
                                    dCldt_dn_dKsn , dCldt_dn_dKse , dCldt_dn_dKsg , dCldt_dn_dKdn , dCldt_dn_dKde , dCldt_dn_dKdg , \
                                    dCldt_dn_dClsn , dCldt_dn_dClse , dCldt_dn_dClsg , dCldt_dn_dCldn , dCldt_dn_dClde , dCldt_dn_dCldg , \
                                    dCldt_dn_dCasn , dCldt_dn_dCase , dCldt_dn_dCadn , dCldt_dn_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dCldt_dn_dVsn , dCldt_dn_dVse , dCldt_dn_dVsg , dCldt_dn_dVdn , dCldt_dn_dVde , dCldt_dn_dVdg]
            dCldt_de_dx = [dCldt_de_dNasn , dCldt_de_dNase , dCldt_de_dNasg , dCldt_de_dNadn , dCldt_de_dNade , dCldt_de_dNadg , \
                                    dCldt_de_dKsn , dCldt_de_dKse , dCldt_de_dKsg , dCldt_de_dKdn , dCldt_de_dKde , dCldt_de_dKdg , \
                                    dCldt_de_dClsn , dCldt_de_dClse , dCldt_de_dClsg , dCldt_de_dCldn , dCldt_de_dClde , dCldt_de_dCldg , \
                                    dCldt_de_dCasn , dCldt_de_dCase , dCldt_de_dCadn , dCldt_de_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dCldt_de_dVsn , dCldt_de_dVse , dCldt_de_dVsg , dCldt_de_dVdn , dCldt_de_dVde , dCldt_de_dVdg]
            dCldt_dg_dx = [dCldt_dg_dNasn , dCldt_dg_dNase , dCldt_dg_dNasg , dCldt_dg_dNadn , dCldt_dg_dNade , dCldt_dg_dNadg , \
                                    dCldt_dg_dKsn , dCldt_dg_dKse , dCldt_dg_dKsg , dCldt_dg_dKdn , dCldt_dg_dKde , dCldt_dg_dKdg , \
                                    dCldt_dg_dClsn , dCldt_dg_dClse , dCldt_dg_dClsg , dCldt_dg_dCldn , dCldt_dg_dClde , dCldt_dg_dCldg , \
                                    dCldt_dg_dCasn , dCldt_dg_dCase , dCldt_dg_dCadn , dCldt_dg_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dCldt_dg_dVsn , dCldt_dg_dVse , dCldt_dg_dVsg , dCldt_dg_dVdn , dCldt_dg_dVde , dCldt_dg_dVdg]
            dCadt_sn_dx = [dCadt_sn_dNasn , dCadt_sn_dNase , dCadt_sn_dNasg , dCadt_sn_dNadn , dCadt_sn_dNade , dCadt_sn_dNadg , \
                                    dCadt_sn_dKsn , dCadt_sn_dKse , dCadt_sn_dKsg , dCadt_sn_dKdn , dCadt_sn_dKde , dCadt_sn_dKdg , \
                                    dCadt_sn_dClsn , dCadt_sn_dClse , dCadt_sn_dClsg , dCadt_sn_dCldn , dCadt_sn_dClde , dCadt_sn_dCldg , \
                                    dCadt_sn_dCasn , dCadt_sn_dCase , dCadt_sn_dCadn , dCadt_sn_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dCadt_sn_dVsn , dCadt_sn_dVse , dCadt_sn_dVsg , dCadt_sn_dVdn , dCadt_sn_dVde , dCadt_sn_dVdg]
            dCadt_se_dx = [dCadt_se_dNasn , dCadt_se_dNase , dCadt_se_dNasg , dCadt_se_dNadn , dCadt_se_dNade , dCadt_se_dNadg , \
                                    dCadt_se_dKsn , dCadt_se_dKse , dCadt_se_dKsg , dCadt_se_dKdn , dCadt_se_dKde , dCadt_se_dKdg , \
                                    dCadt_se_dClsn , dCadt_se_dClse , dCadt_se_dClsg , dCadt_se_dCldn , dCadt_se_dClde , dCadt_se_dCldg , \
                                    dCadt_se_dCasn , dCadt_se_dCase , dCadt_se_dCadn , dCadt_se_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dCadt_se_dVsn , dCadt_se_dVse , dCadt_se_dVsg , dCadt_se_dVdn , dCadt_se_dVde , dCadt_se_dVdg]
            dCadt_dn_dx = [dCadt_dn_dNasn , dCadt_dn_dNase , dCadt_dn_dNasg , dCadt_dn_dNadn , dCadt_dn_dNade , dCadt_dn_dNadg , \
                                    dCadt_dn_dKsn , dCadt_dn_dKse , dCadt_dn_dKsg , dCadt_dn_dKdn , dCadt_dn_dKde , dCadt_dn_dKdg , \
                                    dCadt_dn_dClsn , dCadt_dn_dClse , dCadt_dn_dClsg , dCadt_dn_dCldn , dCadt_dn_dClde , dCadt_dn_dCldg , \
                                    dCadt_dn_dCasn , dCadt_dn_dCase , dCadt_dn_dCadn , dCadt_dn_dCade , \
                                    0 , 0 , dCadt_dn_ds , 0 , 0 , dCadt_dn_dz , \
                                    dCadt_dn_dVsn , dCadt_dn_dVse , dCadt_dn_dVsg , dCadt_dn_dVdn , dCadt_dn_dVde , dCadt_dn_dVdg]
            dCadt_de_dx = [dCadt_de_dNasn , dCadt_de_dNase , dCadt_de_dNasg , dCadt_de_dNadn , dCadt_de_dNade , dCadt_de_dNadg , \
                                    dCadt_de_dKsn , dCadt_de_dKse , dCadt_de_dKsg , dCadt_de_dKdn , dCadt_de_dKde , dCadt_de_dKdg , \
                                    dCadt_de_dClsn , dCadt_de_dClse , dCadt_de_dClsg , dCadt_de_dCldn , dCadt_de_dClde , dCadt_de_dCldg , \
                                    dCadt_de_dCasn , dCadt_de_dCase , dCadt_de_dCadn , dCadt_de_dCade , \
                                    0 , 0 , dCadt_de_ds , 0 , 0 , dCadt_de_dz , \
                                    dCadt_de_dVsn , dCadt_de_dVse , dCadt_de_dVsg , dCadt_de_dVdn , dCadt_de_dVde , dCadt_de_dVdg]
            dndt_dx = [dndt_dNasn , 0 , 0 , 0 , 0 , 0 , \
                                    dndt_dKsn , 0 , 0 , 0 , 0 , 0 , \
                                    dndt_dClsn , 0 , 0 , 0 , 0 , 0 , \
                                    dndt_dCasn , 0 , 0 , 0 , \
                                    dndt_dn , 0 , 0 , 0 , 0 , 0 , \
                                    0 , 0 , 0 , 0 , 0 , 0]
            dhdt_dx = [dhdt_dNasn , 0 , 0 , 0 , 0 , 0 , \
                                    dhdt_dKsn , 0 , 0 , 0 , 0 , 0 , \
                                    dhdt_dClsn , 0 , 0 , 0 , 0 , 0 , \
                                    dhdt_dCasn , 0 , 0 , 0 , \
                                    0 , dhdt_dh , 0 , 0 , 0 , 0 , \
                                    0 , 0 , 0 , 0 , 0 , 0]
            dsdt_dx = [0 , 0 , 0, dsdt_dNadn , 0 , 0 , \
                                    0 , 0 , 0, dsdt_dKdn , 0 , 0 , \
                                    0 , 0 , 0, dsdt_dCldn , 0 , 0 , \
                                    0 , 0, dsdt_dCadn , 0 , \
                                    0 , 0 , dsdt_ds , 0 , 0 , 0 , \
                                    0 , 0 , 0 , 0 , 0 , 0]
            dcdt_dx = [0 , 0 , 0, dcdt_dNadn , 0 , 0 , \
                                    0 , 0 , 0, dcdt_dKdn , 0 , 0 , \
                                    0 , 0 , 0, dcdt_dCldn , 0 , 0 , \
                                    0 , 0, dcdt_dCadn , 0 , \
                                    0 , 0 , 0 , dcdt_dc , 0 , 0 , \
                                    0 , 0 , 0 , 0 , 0 , 0]
            dqdt_dx = [0 , 0 , 0 , 0 , 0 , 0 , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    0 , 0 , dqdt_dCadn , 0 , \
                                    0 , 0 , 0 , 0 , dqdt_dq , 0 , \
                                    0 , 0 , 0 , dqdt_dVdn , 0 , 0 ]
            dzdt_dx = [0 , 0 , 0 , dzdt_dNadn , 0 , 0 , \
                                    0 , 0 , 0 , dzdt_dKdn , 0 , 0 , \
                                    0 , 0 , 0 , dzdt_dCldn , 0 , 0 , \
                                    0 , 0 , dzdt_dCadn , 0 , \
                                    0 , 0 , 0 , 0 , 0 , dzdt_dz , \
                                    0 , 0 , 0 , 0 , 0 , 0 ]

            dVsidt_dx = [dVsidt_dNasn , dVsidt_dNase , 0  , 0 , 0 , 0 , \
                                    dVsidt_dKsn , dVsidt_dKse , 0 , 0 , 0 , 0 , \
                                    dVsidt_dClsn , dVsidt_dClse , 0 , 0 , 0 , 0 , \
                                    dVsidt_dCasn , dVsidt_dCase , 0 , 0 , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dVsidt_dVsn , dVsidt_dVse , 0 , 0 , 0 , 0]
            dVsgdt_dx = [0 , dVsgdt_dNase , dVsgdt_dNasg  , 0 , 0 , 0 , \
                                    0 , dVsgdt_dKse , dVsgdt_dKsg , 0 , 0 , 0 , \
                                    0 , dVsgdt_dClse , dVsgdt_dClsg , 0 , 0 , 0 , \
                                    0 , dVsgdt_dCase , 0 , 0 , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    0 , dVsgdt_dVse , dVsgdt_dVsg , 0 , 0 , 0]
            dVdidt_dx = [0 , 0 , 0 , dVdidt_dNadn , dVdidt_dNade , 0 , \
                                0 , 0 , 0 , dVdidt_dKdn , dVdidt_dKde , 0 , \
                                0 , 0 , 0 , dVdidt_dCldn , dVdidt_dClde , 0 , \
                                0 , 0 , dVdidt_dCadn , dVdidt_dCade , \
                                0 , 0 , 0 , 0 , 0 , 0 , \
                                0 , 0 , 0 , dVdidt_dVdn , dVdidt_dVde , 0 ]
            dVdgdt_dx = [0 , 0 , 0 , 0 , dVdgdt_dNade , dVdgdt_dNadg , \
                                0 , 0 , 0 , 0 , dVdgdt_dKde , dVdgdt_dKdg , \
                                0 , 0 , 0 , 0 , dVdgdt_dClde , dVdgdt_dCldg , \
                                0 , 0 , 0 , dVdgdt_dCade , \
                                0 , 0 , 0 , 0 , 0 , 0 , \
                                0 , 0 , 0 , 0 , dVdgdt_dVde , dVdgdt_dVdg ]
            dVsedt_dx = [dVsedt_dNasn , dVsedt_dNase , dVsedt_dNasg  , 0 , 0 , 0 , \
                                    dVsedt_dKsn , dVsedt_dKse , dVsedt_dKsg , 0 , 0 , 0 , \
                                    dVsedt_dClsn , dVsedt_dClse , dVsedt_dClsg , 0 , 0 , 0 , \
                                    dVsedt_dCasn , dVsedt_dCase , 0 , 0 , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    dVsedt_dVsn , dVsedt_dVse , dVsedt_dVsg , 0 , 0 , 0]
            dVdedt_dx = [0 , 0 , 0 , dVdedt_dNadn , dVdedt_dNade , dVdedt_dNadg , \
                                    0 , 0 , 0 , dVdedt_dKdn , dVdedt_dKde , dVdedt_dKdg , \
                                    0 , 0 , 0 , dVdedt_dCldn , dVdedt_dClde , dVdedt_dCldg , \
                                    0 , 0 , dVdedt_dCadn , dVdedt_dCade , \
                                    0 , 0 , 0 , 0 , 0 , 0 , \
                                    0 , 0 , 0 , dVdedt_dVdn , dVdedt_dVde , dVdedt_dVdg ]
        
            return [dNadt_sn_dx, dNadt_se_dx, dNadt_sg_dx, dNadt_dn_dx, dNadt_de_dx, dNadt_dg_dx, dKdt_sn_dx, dKdt_se_dx, dKdt_sg_dx, dKdt_dn_dx, dKdt_de_dx, dKdt_dg_dx, \
                dCldt_sn_dx, dCldt_se_dx, dCldt_sg_dx, dCldt_dn_dx, dCldt_de_dx, dCldt_dg_dx, dCadt_sn_dx, dCadt_se_dx, dCadt_dn_dx, dCadt_de_dx, \
                dndt_dx, dhdt_dx, dsdt_dx, dcdt_dx, dqdt_dx, dzdt_dx, dVsidt_dx, dVsedt_dx, dVsgdt_dx, dVdidt_dx, dVdedt_dx, dVdgdt_dx]
        
        # Build sparse Jacobian matrix
        else:
            row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,              #dNadt
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,

                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,                          #dKdt 
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                
                12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, #dCldt
                13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
                15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,

                18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, #dCadt
                19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,

                22, 22, 22, 22, 22,     #dndt
                23, 23, 23, 23, 23,     #dhdt
                24, 24, 24, 24, 24,     #dsdt
                25, 25, 25, 25, 25,     #dcdt

                26, 26, 26,             #dqdt
                27, 27, 27, 27, 27,     #dzdt

                28, 28, 28, 28, 28, 28, 28, 28, 28, 28,                 #dVsidt
                29, 29, 29, 29, 29, 29, 29, 29, 29,                     #dVsgdt
                30, 30, 30, 30, 30, 30, 30, 30, 30, 30,                 #dVdidt
                31, 31, 31, 31, 31, 31, 31, 31, 31,                     #dVdgdt
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, #dVsedt
                33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33  #dVdedt
                ])

            col = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 28, 29, 30, 31, 32, 33, #dNadt
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 28, 29, 30, 31, 32, 33, 
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33, 
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33, 
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33, 
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33, 

                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28, 29, 30, 31, 32, 33,               #dKdt
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28, 29, 30, 31, 32, 33,  
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 28, 29, 30, 31, 32, 33,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,

                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,                   #dCldt
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,

                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,                   #dCadt
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 28, 29, 30, 31, 32, 33,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 28, 29, 30, 31, 32, 33,

                0, 6, 12, 18, 22,   #dndt
                0, 6, 12, 18, 23,   #dhdt
                3, 9, 15, 20, 24,   #dsdt
                3, 9, 15, 20, 25,   #dcdt

                20, 26, 31,         #dqdt
                3, 9, 15, 20, 27,   #dzdt

                0, 1, 6, 7, 12, 13, 18, 19, 28, 29,                 #dVsidt
                1, 2, 7, 8, 13, 14, 19, 29, 30,                     #dVsgdt
                3, 4, 9, 10, 15, 16, 20, 21, 31, 32,                #dVdidt
                4, 5, 10, 11, 16, 17, 21, 32, 33,                   #dVdgdt

                0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 28, 29, 30,   #dVsedt
                3, 4, 5, 9, 10, 11, 15, 16, 17, 20, 21, 31, 32, 33  #dVdedt
                ])
            data = np.array([ dNadt_sn_dNasn , dNadt_sn_dNase , dNadt_sn_dNasg , dNadt_sn_dNadn , dNadt_sn_dNade , dNadt_sn_dNadg ,
                dNadt_sn_dKsn , dNadt_sn_dKse , dNadt_sn_dKsg , dNadt_sn_dKdn , dNadt_sn_dKde , dNadt_sn_dKdg ,
                dNadt_sn_dClsn , dNadt_sn_dClse , dNadt_sn_dClsg , dNadt_sn_dCldn , dNadt_sn_dClde , dNadt_sn_dCldg ,
                dNadt_sn_dCasn , dNadt_sn_dCase , dNadt_sn_dCadn , dNadt_sn_dCade ,
                dNadt_sn_dh ,
                dNadt_sn_dVsn , dNadt_sn_dVse , dNadt_sn_dVsg , dNadt_sn_dVdn , dNadt_sn_dVde , dNadt_sn_dVdg,
                dNadt_se_dNasn , dNadt_se_dNase , dNadt_se_dNasg , dNadt_se_dNadn , dNadt_se_dNade , dNadt_se_dNadg ,
                dNadt_se_dKsn , dNadt_se_dKse , dNadt_se_dKsg , dNadt_se_dKdn , dNadt_se_dKde , dNadt_se_dKdg ,
                dNadt_se_dClsn , dNadt_se_dClse , dNadt_se_dClsg , dNadt_se_dCldn , dNadt_se_dClde , dNadt_se_dCldg ,
                dNadt_se_dCasn , dNadt_se_dCase , dNadt_se_dCadn , dNadt_se_dCade ,
                dNadt_se_dh ,    
                dNadt_se_dVsn , dNadt_se_dVse , dNadt_se_dVsg , dNadt_se_dVdn , dNadt_se_dVde , dNadt_se_dVdg,
                dNadt_sg_dNasn , dNadt_sg_dNase , dNadt_sg_dNasg , dNadt_sg_dNadn , dNadt_sg_dNade , dNadt_sg_dNadg ,
                dNadt_sg_dKsn , dNadt_sg_dKse , dNadt_sg_dKsg , dNadt_sg_dKdn , dNadt_sg_dKde , dNadt_sg_dKdg ,
                dNadt_sg_dClsn , dNadt_sg_dClse , dNadt_sg_dClsg , dNadt_sg_dCldn , dNadt_sg_dClde , dNadt_sg_dCldg ,
                dNadt_sg_dCasn , dNadt_sg_dCase , dNadt_sg_dCadn , dNadt_sg_dCade ,          
                dNadt_sg_dVsn , dNadt_sg_dVse , dNadt_sg_dVsg , dNadt_sg_dVdn , dNadt_sg_dVde , dNadt_sg_dVdg,
                dNadt_dn_dNasn , dNadt_dn_dNase , dNadt_dn_dNasg , dNadt_dn_dNadn , dNadt_dn_dNade , dNadt_dn_dNadg ,
                dNadt_dn_dKsn , dNadt_dn_dKse , dNadt_dn_dKsg , dNadt_dn_dKdn , dNadt_dn_dKde , dNadt_dn_dKdg ,
                dNadt_dn_dClsn , dNadt_dn_dClse , dNadt_dn_dClsg , dNadt_dn_dCldn , dNadt_dn_dClde , dNadt_dn_dCldg ,
                dNadt_dn_dCasn , dNadt_dn_dCase , dNadt_dn_dCadn , dNadt_dn_dCade ,                             
                dNadt_dn_dVsn , dNadt_dn_dVse , dNadt_dn_dVsg , dNadt_dn_dVdn , dNadt_dn_dVde , dNadt_dn_dVdg,
                dNadt_de_dNasn , dNadt_de_dNase , dNadt_de_dNasg , dNadt_de_dNadn , dNadt_de_dNade , dNadt_de_dNadg ,
                dNadt_de_dKsn , dNadt_de_dKse , dNadt_de_dKsg , dNadt_de_dKdn , dNadt_de_dKde , dNadt_de_dKdg ,
                dNadt_de_dClsn , dNadt_de_dClse , dNadt_de_dClsg , dNadt_de_dCldn , dNadt_de_dClde , dNadt_de_dCldg ,
                dNadt_de_dCasn , dNadt_de_dCase , dNadt_de_dCadn , dNadt_de_dCade ,                             
                dNadt_de_dVsn , dNadt_de_dVse , dNadt_de_dVsg , dNadt_de_dVdn , dNadt_de_dVde , dNadt_de_dVdg,
                dNadt_dg_dNasn , dNadt_dg_dNase , dNadt_dg_dNasg , dNadt_dg_dNadn , dNadt_dg_dNade , dNadt_dg_dNadg ,
                dNadt_dg_dKsn , dNadt_dg_dKse , dNadt_dg_dKsg , dNadt_dg_dKdn , dNadt_dg_dKde , dNadt_dg_dKdg ,
                dNadt_dg_dClsn , dNadt_dg_dClse , dNadt_dg_dClsg , dNadt_dg_dCldn , dNadt_dg_dClde , dNadt_dg_dCldg ,
                dNadt_dg_dCasn , dNadt_dg_dCase , dNadt_dg_dCadn , dNadt_dg_dCade ,                             
                dNadt_dg_dVsn , dNadt_dg_dVse , dNadt_dg_dVsg , dNadt_dg_dVdn , dNadt_dg_dVde , dNadt_dg_dVdg,
                dKdt_sn_dNasn , dKdt_sn_dNase , dKdt_sn_dNasg , dKdt_sn_dNadn , dKdt_sn_dNade , dKdt_sn_dNadg ,
                dKdt_sn_dKsn , dKdt_sn_dKse , dKdt_sn_dKsg , dKdt_sn_dKdn , dKdt_sn_dKde , dKdt_sn_dKdg ,
                dKdt_sn_dClsn , dKdt_sn_dClse , dKdt_sn_dClsg , dKdt_sn_dCldn , dKdt_sn_dClde , dKdt_sn_dCldg ,
                dKdt_sn_dCasn , dKdt_sn_dCase , dKdt_sn_dCadn , dKdt_sn_dCade ,
                dKdt_sn_dn ,     
                dKdt_sn_dVsn , dKdt_sn_dVse , dKdt_sn_dVsg , dKdt_sn_dVdn , dKdt_sn_dVde , dKdt_sn_dVdg,
                dKdt_se_dNasn , dKdt_se_dNase , dKdt_se_dNasg , dKdt_se_dNadn , dKdt_se_dNade , dKdt_se_dNadg ,
                dKdt_se_dKsn , dKdt_se_dKse , dKdt_se_dKsg , dKdt_se_dKdn , dKdt_se_dKde , dKdt_se_dKdg ,
                dKdt_se_dClsn , dKdt_se_dClse , dKdt_se_dClsg , dKdt_se_dCldn , dKdt_se_dClde , dKdt_se_dCldg ,
                dKdt_se_dCasn , dKdt_se_dCase , dKdt_se_dCadn , dKdt_se_dCade ,
                dKdt_se_dn ,     
                dKdt_se_dVsn , dKdt_se_dVse , dKdt_se_dVsg , dKdt_se_dVdn , dKdt_se_dVde , dKdt_se_dVdg,
                dKdt_sg_dNasn , dKdt_sg_dNase , dKdt_sg_dNasg , dKdt_sg_dNadn , dKdt_sg_dNade , dKdt_sg_dNadg ,
                dKdt_sg_dKsn , dKdt_sg_dKse , dKdt_sg_dKsg , dKdt_sg_dKdn , dKdt_sg_dKde , dKdt_sg_dKdg ,
                dKdt_sg_dClsn , dKdt_sg_dClse , dKdt_sg_dClsg , dKdt_sg_dCldn , dKdt_sg_dClde , dKdt_sg_dCldg ,
                dKdt_sg_dCasn , dKdt_sg_dCase , dKdt_sg_dCadn , dKdt_sg_dCade ,                             
                dKdt_sg_dVsn , dKdt_sg_dVse , dKdt_sg_dVsg , dKdt_sg_dVdn , dKdt_sg_dVde , dKdt_sg_dVdg, 
                dKdt_dn_dNasn , dKdt_dn_dNase , dKdt_dn_dNasg , dKdt_dn_dNadn , dKdt_dn_dNade , dKdt_dn_dNadg ,
                dKdt_dn_dKsn , dKdt_dn_dKse , dKdt_dn_dKsg , dKdt_dn_dKdn , dKdt_dn_dKde , dKdt_dn_dKdg ,
                dKdt_dn_dClsn , dKdt_dn_dClse , dKdt_dn_dClsg , dKdt_dn_dCldn , dKdt_dn_dClde , dKdt_dn_dCldg ,
                dKdt_dn_dCasn , dKdt_dn_dCase , dKdt_dn_dCadn , dKdt_dn_dCade ,
                dKdt_dn_dc , dKdt_dn_dq , 
                dKdt_dn_dVsn , dKdt_dn_dVse , dKdt_dn_dVsg , dKdt_dn_dVdn , dKdt_dn_dVde , dKdt_dn_dVdg,
                dKdt_de_dNasn , dKdt_de_dNase , dKdt_de_dNasg , dKdt_de_dNadn , dKdt_de_dNade , dKdt_de_dNadg ,
                dKdt_de_dKsn , dKdt_de_dKse , dKdt_de_dKsg , dKdt_de_dKdn , dKdt_de_dKde , dKdt_de_dKdg ,
                dKdt_de_dClsn , dKdt_de_dClse , dKdt_de_dClsg , dKdt_de_dCldn , dKdt_de_dClde , dKdt_de_dCldg ,
                dKdt_de_dCasn , dKdt_de_dCase , dKdt_de_dCadn , dKdt_de_dCade ,                             
                dKdt_de_dVsn , dKdt_de_dVse , dKdt_de_dVsg , dKdt_de_dVdn , dKdt_de_dVde , dKdt_de_dVdg,
                dKdt_dg_dNasn , dKdt_dg_dNase , dKdt_dg_dNasg , dKdt_dg_dNadn , dKdt_dg_dNade , dKdt_dg_dNadg ,
                dKdt_dg_dKsn , dKdt_dg_dKse , dKdt_dg_dKsg , dKdt_dg_dKdn , dKdt_dg_dKde , dKdt_dg_dKdg ,
                dKdt_dg_dClsn , dKdt_dg_dClse , dKdt_dg_dClsg , dKdt_dg_dCldn , dKdt_dg_dClde , dKdt_dg_dCldg ,
                dKdt_dg_dCasn , dKdt_dg_dCase , dKdt_dg_dCadn , dKdt_dg_dCade ,                             
                dKdt_dg_dVsn , dKdt_dg_dVse , dKdt_dg_dVsg , dKdt_dg_dVdn , dKdt_dg_dVde , dKdt_dg_dVdg,
                dCldt_sn_dNasn , dCldt_sn_dNase , dCldt_sn_dNasg , dCldt_sn_dNadn , dCldt_sn_dNade , dCldt_sn_dNadg ,
                dCldt_sn_dKsn , dCldt_sn_dKse , dCldt_sn_dKsg , dCldt_sn_dKdn , dCldt_sn_dKde , dCldt_sn_dKdg ,
                dCldt_sn_dClsn , dCldt_sn_dClse , dCldt_sn_dClsg , dCldt_sn_dCldn , dCldt_sn_dClde , dCldt_sn_dCldg ,
                dCldt_sn_dCasn , dCldt_sn_dCase , dCldt_sn_dCadn , dCldt_sn_dCade ,                             
                dCldt_sn_dVsn , dCldt_sn_dVse , dCldt_sn_dVsg , dCldt_sn_dVdn , dCldt_sn_dVde , dCldt_sn_dVdg,
                dCldt_se_dNasn , dCldt_se_dNase , dCldt_se_dNasg , dCldt_se_dNadn , dCldt_se_dNade , dCldt_se_dNadg ,
                dCldt_se_dKsn , dCldt_se_dKse , dCldt_se_dKsg , dCldt_se_dKdn , dCldt_se_dKde , dCldt_se_dKdg ,
                dCldt_se_dClsn , dCldt_se_dClse , dCldt_se_dClsg , dCldt_se_dCldn , dCldt_se_dClde , dCldt_se_dCldg ,
                dCldt_se_dCasn , dCldt_se_dCase , dCldt_se_dCadn , dCldt_se_dCade ,                             
                dCldt_se_dVsn , dCldt_se_dVse , dCldt_se_dVsg , dCldt_se_dVdn , dCldt_se_dVde , dCldt_se_dVdg,
                dCldt_sg_dNasn , dCldt_sg_dNase , dCldt_sg_dNasg , dCldt_sg_dNadn , dCldt_sg_dNade , dCldt_sg_dNadg ,
                dCldt_sg_dKsn , dCldt_sg_dKse , dCldt_sg_dKsg , dCldt_sg_dKdn , dCldt_sg_dKde , dCldt_sg_dKdg ,
                dCldt_sg_dClsn , dCldt_sg_dClse , dCldt_sg_dClsg , dCldt_sg_dCldn , dCldt_sg_dClde , dCldt_sg_dCldg ,
                dCldt_sg_dCasn , dCldt_sg_dCase , dCldt_sg_dCadn , dCldt_sg_dCade ,                             
                dCldt_sg_dVsn , dCldt_sg_dVse , dCldt_sg_dVsg , dCldt_sg_dVdn , dCldt_sg_dVde , dCldt_sg_dVdg,
                dCldt_dn_dNasn , dCldt_dn_dNase , dCldt_dn_dNasg , dCldt_dn_dNadn , dCldt_dn_dNade , dCldt_dn_dNadg ,
                dCldt_dn_dKsn , dCldt_dn_dKse , dCldt_dn_dKsg , dCldt_dn_dKdn , dCldt_dn_dKde , dCldt_dn_dKdg ,
                dCldt_dn_dClsn , dCldt_dn_dClse , dCldt_dn_dClsg , dCldt_dn_dCldn , dCldt_dn_dClde , dCldt_dn_dCldg ,
                dCldt_dn_dCasn , dCldt_dn_dCase , dCldt_dn_dCadn , dCldt_dn_dCade ,                             
                dCldt_dn_dVsn , dCldt_dn_dVse , dCldt_dn_dVsg , dCldt_dn_dVdn , dCldt_dn_dVde , dCldt_dn_dVdg,
                dCldt_de_dNasn , dCldt_de_dNase , dCldt_de_dNasg , dCldt_de_dNadn , dCldt_de_dNade , dCldt_de_dNadg ,
                dCldt_de_dKsn , dCldt_de_dKse , dCldt_de_dKsg , dCldt_de_dKdn , dCldt_de_dKde , dCldt_de_dKdg ,
                dCldt_de_dClsn , dCldt_de_dClse , dCldt_de_dClsg , dCldt_de_dCldn , dCldt_de_dClde , dCldt_de_dCldg ,
                dCldt_de_dCasn , dCldt_de_dCase , dCldt_de_dCadn , dCldt_de_dCade ,                             
                dCldt_de_dVsn , dCldt_de_dVse , dCldt_de_dVsg , dCldt_de_dVdn , dCldt_de_dVde , dCldt_de_dVdg,
                dCldt_dg_dNasn , dCldt_dg_dNase , dCldt_dg_dNasg , dCldt_dg_dNadn , dCldt_dg_dNade , dCldt_dg_dNadg ,
                dCldt_dg_dKsn , dCldt_dg_dKse , dCldt_dg_dKsg , dCldt_dg_dKdn , dCldt_dg_dKde , dCldt_dg_dKdg ,
                dCldt_dg_dClsn , dCldt_dg_dClse , dCldt_dg_dClsg , dCldt_dg_dCldn , dCldt_dg_dClde , dCldt_dg_dCldg ,
                dCldt_dg_dCasn , dCldt_dg_dCase , dCldt_dg_dCadn , dCldt_dg_dCade ,                             
                dCldt_dg_dVsn , dCldt_dg_dVse , dCldt_dg_dVsg , dCldt_dg_dVdn , dCldt_dg_dVde , dCldt_dg_dVdg,
                dCadt_sn_dNasn , dCadt_sn_dNase , dCadt_sn_dNasg , dCadt_sn_dNadn , dCadt_sn_dNade , dCadt_sn_dNadg ,
                dCadt_sn_dKsn , dCadt_sn_dKse , dCadt_sn_dKsg , dCadt_sn_dKdn , dCadt_sn_dKde , dCadt_sn_dKdg ,
                dCadt_sn_dClsn , dCadt_sn_dClse , dCadt_sn_dClsg , dCadt_sn_dCldn , dCadt_sn_dClde , dCadt_sn_dCldg ,
                dCadt_sn_dCasn , dCadt_sn_dCase , dCadt_sn_dCadn , dCadt_sn_dCade ,                             
                dCadt_sn_dVsn , dCadt_sn_dVse , dCadt_sn_dVsg , dCadt_sn_dVdn , dCadt_sn_dVde , dCadt_sn_dVdg,
                dCadt_se_dNasn , dCadt_se_dNase , dCadt_se_dNasg , dCadt_se_dNadn , dCadt_se_dNade , dCadt_se_dNadg ,
                dCadt_se_dKsn , dCadt_se_dKse , dCadt_se_dKsg , dCadt_se_dKdn , dCadt_se_dKde , dCadt_se_dKdg ,
                dCadt_se_dClsn , dCadt_se_dClse , dCadt_se_dClsg , dCadt_se_dCldn , dCadt_se_dClde , dCadt_se_dCldg ,
                dCadt_se_dCasn , dCadt_se_dCase , dCadt_se_dCadn , dCadt_se_dCade ,                             
                dCadt_se_dVsn , dCadt_se_dVse , dCadt_se_dVsg , dCadt_se_dVdn , dCadt_se_dVde , dCadt_se_dVdg,
                dCadt_dn_dNasn , dCadt_dn_dNase , dCadt_dn_dNasg , dCadt_dn_dNadn , dCadt_dn_dNade , dCadt_dn_dNadg ,
                dCadt_dn_dKsn , dCadt_dn_dKse , dCadt_dn_dKsg , dCadt_dn_dKdn , dCadt_dn_dKde , dCadt_dn_dKdg ,
                dCadt_dn_dClsn , dCadt_dn_dClse , dCadt_dn_dClsg , dCadt_dn_dCldn , dCadt_dn_dClde , dCadt_dn_dCldg ,
                dCadt_dn_dCasn , dCadt_dn_dCase , dCadt_dn_dCadn , dCadt_dn_dCade ,
                dCadt_dn_ds ,   dCadt_dn_dz ,
                dCadt_dn_dVsn , dCadt_dn_dVse , dCadt_dn_dVsg , dCadt_dn_dVdn , dCadt_dn_dVde , dCadt_dn_dVdg,
                dCadt_de_dNasn , dCadt_de_dNase , dCadt_de_dNasg , dCadt_de_dNadn , dCadt_de_dNade , dCadt_de_dNadg ,
                dCadt_de_dKsn , dCadt_de_dKse , dCadt_de_dKsg , dCadt_de_dKdn , dCadt_de_dKde , dCadt_de_dKdg ,
                dCadt_de_dClsn , dCadt_de_dClse , dCadt_de_dClsg , dCadt_de_dCldn , dCadt_de_dClde , dCadt_de_dCldg ,
                dCadt_de_dCasn , dCadt_de_dCase , dCadt_de_dCadn , dCadt_de_dCade ,
                dCadt_de_ds ,   dCadt_de_dz ,
                dCadt_de_dVsn , dCadt_de_dVse , dCadt_de_dVsg , dCadt_de_dVdn , dCadt_de_dVde , dCadt_de_dVdg,
                
                dndt_dNasn , dndt_dKsn , dndt_dClsn , dndt_dCasn , dndt_dn ,
                dhdt_dNasn , dhdt_dKsn , dhdt_dClsn , dhdt_dCasn , dhdt_dh ,
                dsdt_dNadn , dsdt_dKdn , dsdt_dCldn , dsdt_dCadn , dsdt_ds ,
                dcdt_dNadn , dcdt_dKdn , dcdt_dCldn , dcdt_dCadn , dcdt_dc , 
                
                dqdt_dCadn , dqdt_dq , dqdt_dVdn,
                dzdt_dNadn , dzdt_dKdn ,dzdt_dCldn , dzdt_dCadn , dzdt_dz ,
                    
                dVsidt_dNasn , dVsidt_dNase ,  dVsidt_dKsn , dVsidt_dKse ,  dVsidt_dClsn , dVsidt_dClse ,    
                dVsidt_dCasn , dVsidt_dCase ,  dVsidt_dVsn , dVsidt_dVse ,    
                
                dVsgdt_dNase , dVsgdt_dNasg ,  dVsgdt_dKse , dVsgdt_dKsg ,  dVsgdt_dClse , dVsgdt_dClsg ,   
                dVsgdt_dCase , dVsgdt_dVse ,  dVsgdt_dVsg ,   
                
                dVdidt_dNadn , dVdidt_dNade , dVdidt_dKdn , dVdidt_dKde , dVdidt_dCldn , dVdidt_dClde , 
                dVdidt_dCadn , dVdidt_dCade , dVdidt_dVdn , dVdidt_dVde , 
                
                dVdgdt_dNade , dVdgdt_dNadg , dVdgdt_dKde , dVdgdt_dKdg , dVdgdt_dClde , dVdgdt_dCldg ,
                dVdgdt_dCade , dVdgdt_dVde , dVdgdt_dVdg ,

                dVsedt_dNasn , dVsedt_dNase , dVsedt_dNasg , dVsedt_dKsn , dVsedt_dKse , dVsedt_dKsg ,   
                dVsedt_dClsn , dVsedt_dClse , dVsedt_dClsg , dVsedt_dCasn , dVsedt_dCase ,                             
                dVsedt_dVsn , dVsedt_dVse , dVsedt_dVsg ,  

                dVdedt_dNadn , dVdedt_dNade , dVdedt_dNadg , dVdedt_dKdn , dVdedt_dKde , dVdedt_dKdg ,
                dVdedt_dCldn , dVdedt_dClde , dVdedt_dCldg , dVdedt_dCadn , dVdedt_dCade ,
                dVdedt_dVdn , dVdedt_dVde , dVdedt_dVdg 
                ])
            
            return csr_matrix((data, (row,col)), shape=(34,34))