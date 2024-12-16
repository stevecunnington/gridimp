import numpy as np
from scipy.interpolate import interp1d

def SetCosmology(builtincosmo='Planck18',z=0,UseCLASS=True,UseCAMB=False):
    '''Use to initialise cosmology kernel'''
    if UseCAMB==True:
        UseCLASS = False
        import camb
    if UseCLASS==True: import classylss
    global H_0
    global h
    global Om0
    global Ob0
    global n_s
    global A_s
    global delta_c
    if builtincosmo=='Planck15':
        Om0 = 0.307
        Ob0 = 0.0486 # Omega_b
        n_s = 0.968
    if builtincosmo=='Planck18':
        Om0 = 0.315
        Ob0 = 0.0489 # Omega_b
        n_s = 0.965
    global cosmo_
    if builtincosmo=='Planck15': from astropy.cosmology import Planck15 as cosmo_
    if builtincosmo=='Planck18': from astropy.cosmology import Planck18 as cosmo_
    H_0 = cosmo_.H(0).value
    h = H_0/100
    A_s = 2.14e-9 # Scalar amplitude

def f(z):
    gamma = 0.545
    return Omega_m(z)**gamma

def H(z):
    return cosmo_.H(z).value

def Omega_m(z):
    return H_0**2*Om0*(1+z)**3 / H(z)**2

def d_com(z):
    '''Comoving distance (using astropy) in Mpc/h'''
    return cosmo_.comoving_distance(z).value * h

def GetModelPk(z,kmin=1e-3,kmax=10,NonLinear=False,UseCLASS=True,UseCAMB=False):
    '''Generate model matter power spectrum at redshift z using classy'''
    if UseCAMB==True:
        UseCLASS = False
        import camb
        from camb import model, initialpower
        # Declare minium k value for avoiding interpolating outside this value
        global kmin_interp
        kmin_interp = kmin
        Oc0 = Om0 - Ob0 # Omega_c
        #Set up the fiducial cosmology
        pars = camb.CAMBparams()
        #Set cosmology
        pars.set_cosmology(H0=H_0,ombh2=Ob0*h**2,omch2=Oc0*h**2,omk=0,mnu=0)
        pars.set_dark_energy() #LCDM (default)
        pars.InitPower.set_params(ns=n_s, r=0, As=A_s)
        pars.set_for_lmax(2500, lens_potential_accuracy=0);
        #Calculate results for these parameters
        results = camb.get_results(pars)
        #Get matter power spectrum at some redshift
        pars.set_matter_power(redshifts=[z], kmax=kmax)
        if NonLinear==False: pars.NonLinear = model.NonLinear_none
        if NonLinear==True: pars.NonLinear = model.NonLinear_both # Uses HaloFit
        results.calc_power_spectra(pars)
        k, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = 200)
        # Define global transfer function to be called in other functions:
        trans = results.get_matter_transfer_data()
        k_trans = trans.transfer_data[0,:,0] #get kh - the values of k/h at which transfer function is calculated
        transfer_func = trans.transfer_data[model.Transfer_cdm-1,:,0]
        transfer_func = transfer_func/np.max(transfer_func)
        global T
        T = interp1d(k_trans, transfer_func) # Transfer function - set to global variable
        return interp1d(k, pk[0])
    if UseCLASS==True:
        import classylss.binding as CLASS
        if NonLinear==False: cosmo = CLASS.ClassEngine({'output': 'dTk vTk mPk', 'P_k_max_h/Mpc' : kmax, "z_max_pk" : 100.0})
        if NonLinear==True: cosmo = CLASS.ClassEngine({'output': 'dTk vTk mPk', 'non linear': 'halofit', 'P_k_max_h/Mpc' : kmax, "z_max_pk" : 100.0})
        sp = CLASS.Spectra(cosmo)
        k = np.linspace(kmin,kmax,10000)
        return interp1d(k, sp.get_pk(k=k,z=z) )
