import numpy as np
import cosmo
import line
from scipy.ndimage import gaussian_filter
c = 299792458 # speed of light m/s

def getbeampars(D_dish,nu,gamma=None,verbose=False):
    # Return beam size for given dish-size and frequency in MHz
    d_c = cosmo.d_com( line.nu21cm_to_z(nu)) # Comoving distance to frequency bin
    theta_FWHM = np.degrees(c / (nu*1e6 * D_dish)) # freq-dependent beam size
    sig_beam = theta_FWHM/(2*np.sqrt(2*np.log(2)))
    R_beam = d_c * np.radians(sig_beam) #Beam sigma
    if gamma is not None and gamma!=0: R_beam = gamma*R_beam
    if verbose==True: print('\nTelescope params: Dish size =',D_dish,'m, R_beam =',np.round(R_beam,1),'Mpc/h, theta_FWHM =',np.round(theta_FWHM,2),'deg')
    return theta_FWHM,R_beam

def gaussbeam(map,R_beam,dims):
    '''Smooth map in angular direction with Gaussian kernel to represent approximate
    beam. R_beam is the sigma of beam profile in [Mpc/h]'''
    lx,ly,lz,nx,ny,nz = dims[:6]
    dpix = np.mean([lx/nx,ly/ny])
    for iz in range(nz):
        map[:,:,iz] = gaussian_filter(map[:,:,iz],sigma=R_beam/dpix,mode='wrap')
    return map

def P_noise(A_sky,theta_FWHM,t_tot,N_dish,nu,lz,T_sys=None,deltav=1,epsilon=1,hitmap=None,return_sigma_N=False,verbose=False):
    ### Return a scale invariant level for the thermal noise power spectrum
    # To ensure a correct forecast, the pixel volumes used to go from sigma_N to
    #   P_N are based on the given beam size and freq resolution
    '''
    Based on Santos+15 (https://arxiv.org/pdf/1501.03989.pdf) eq 5.1
     - theta_FWHM beam size to base pixel size on (use minimum beam size) should be the same
        for all frequencies since angular pixel size will be the same at all frequencies
    '''
    if T_sys is None: # Calculate based on SKA red book eq1: https://arxiv.org/pdf/1811.02743.pdf
        Tspl = 3e3 #mK
        TCMB = 2.73e3 #mk
        T_sys = np.zeros(len(nu))
        for i in range(len(nu)):
            Tgal = 25e3*(408/nu[i])**2.75
            #Trx = 15e3 + 30e3*(nu[i]/1e3 - 0.75)**2 # From Red Book
            Trx = 7.5e3 + 10e3*(nu[i]/1e3 - 0.75)**2 # Amended from above to better fit Wang+20 MK Pilot Survey
            T_sys[i] = Trx + Tspl + TCMB + Tgal
        if verbose==True: print('\nCalculated System Temp [K]: %s'%np.round(np.min(T_sys)/1e3,2),'< T_sys < %s'%np.round(np.max(T_sys)/1e3,2) )
    else: T_sys = np.repeat(T_sys,len(nu)) # For freq independendent given T_sys
    deltav = deltav * 1e6 # Convert MHz to Hz
    t_tot = t_tot * 60 * 60 # Convert observing hours to seconds
    pix_size = theta_FWHM / 3 # [deg] based on MeerKAT pilot survey approach
    A_p = pix_size**2 # Area covered in each pointing (related to beam size - equation formed by Steve)
    N_p = A_sky / A_p # Number of pointings
    t_p = N_dish * t_tot / N_p  # time per pointing
    sigma_N = T_sys / (epsilon * np.sqrt(2 * deltav * t_p) ) # Santos+15 eq 5.1
    if return_sigma_N==True: return sigma_N
    nchannels = (np.max(nu) - np.min(nu))*1e6 / deltav # Effective number of channels given freq resolution
    deltalz = lz/nchannels # [Mpc/h] depth of each voxel on grid
    P_N = np.zeros(len(nu))
    for i in range(len(nu)):
        z = line.nu21cm_to_z(nu[i])
        d_c = cosmo.d_com(z)
        pix_area = (d_c * np.radians(pix_size) )**2 # [Mpc/h]^2 based on fixed pixels size in deg
        V_cell = pix_area * deltalz
        P_N[i] = V_cell * sigma_N[i]**2
    return P_N
