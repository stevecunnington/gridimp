import numpy as np
nu_21cm = 1420.405751 #21cm frequency in MHz at z=0

def z_to_nu21cm(z):
    '''Convert redshift to frequency for HI emission (freq in MHz)'''
    return nu_21cm / (1+z)

def nu21cm_to_z(nu):
    '''Convert frequency to redshift for HI emission (freq in MHz)'''
    return (nu_21cm/nu) - 1

def T_21cm(z,OmegaHI):
    # Mean brightness temperature for HI - Battye+13 formula
    from gridimp import cosmo
    H0 = cosmo.H(0)
    Hz = cosmo.H(z) #km / Mpc s
    h = H0/100
    return 180 * OmegaHI * h * (1+z)**2 / (Hz/H0)

def b_21cm(z):
    '''
    Use 6 values for HI bias at redshifts 0 to 5 found in Table 5 of
    Villaescusa-Navarro et al.(2018) https://arxiv.org/pdf/1804.09180.pdf
    and get a polyfit function based on these values
    '''
    #### Code for finding polynomial coeficients: #####
    #z = np.array([0,1,2,3,4,5])
    #b_HI = np.array([0.84, 1.49, 2.03, 2.56, 2.82, 3.18])
    #coef = np.polyfit(z, b_HI,2)
    #A,B,C = coef[2],coef[1],coef[0]
    ###################################################
    A,B,C = 0.84178571,0.69289286,-0.04589286
    return A + B*z + C*z**2
