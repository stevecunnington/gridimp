import numpy as np
from scipy.interpolate import interp1d

def SetCosmology(builtincosmo='Planck18',z=0):
    '''Use to initialise cosmology kernel'''
    import classylss
    global H_0
    global h
    global Om0
    if builtincosmo=='Planck15': Om0 = 0.307
    if builtincosmo=='Planck18': Om0 = 0.315
    global cosmo_
    if builtincosmo=='Planck15': from astropy.cosmology import Planck15 as cosmo_
    if builtincosmo=='Planck18': from astropy.cosmology import Planck18 as cosmo_
    H_0 = cosmo_.H(0).value
    h = H_0/100

def f(z):
    gamma = 0.545
    return Omega_m(z)**gamma

def H(z):
    return cosmo_.H(z).value

def Omega_m(z):
    return H_0**2*Om0*(1+z)**3 / H(z)**2

def d_com(z,UseCamb=False):
    '''Comoving distance (using astropy) in Mpc/h'''
    return cosmo_.comoving_distance(z).value * h

def GetModelPk(z,kmin=1e-3,kmax=10,NonLinear=False):
    '''Generate model matter power spectrum at redshift z using classy'''
    import classylss.binding as CLASS
    if NonLinear==False: cosmo = CLASS.ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc' : kmax, "z_max_pk" : 100.0})
    if NonLinear==True: cosmo = CLASS.ClassEngine({'output': 'dTk vTk mPk', 'non linear': 'halofit', 'P_k_max_h/Mpc' : kmax, "z_max_pk" : 100.0})
    sp = CLASS.Spectra(cosmo)
    k = np.linspace(kmin,kmax,10000)
    return interp1d(k, sp.get_pk(k=k,z=z) )
