import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

##### CHANGE THIS TO PIP INSTALL GRIDIMP ##############
import sys
sys.path.insert(1, '/Users/user/Documents/gridimp/gridimp')
sys.path.insert(1, '/users/scunnington/gridimp/gridimp')

from astropy_healpix import HEALPix
from astropy import units as u
from scipy.ndimage import gaussian_filter

def init(survey='Initial',dobeam=True,dohealpy=True,fft2hp_ratio=2):
    #survey: Default or Wide

    ### Define an arbitrarty survey size in RA/Dec with a frequency range:
    # - also chose number of cells for input grid size and final FFT grid
    if survey=='Initial': # survey with ~cubical voxels/cells
        ramin,ramax = 10,30
        decmin,decmax = 10,30.1 #[need 30.1 otherwise differences on Ilifu and local ?????]
        numin,numax = 925.5,1063.5
        nnu = 118 # number of frequency channels
    if survey=='FineChannel':
        ramin,ramax = 10,30
        decmin,decmax = 10,30.1 #[need 30.1 otherwise differences on Ilifu and local ?????]
        numin,numax = 900,1100
        nnu = 400 # number of frequency channels
    if survey=='WideSky':
        ramin,ramax = 10,40
        decmin,decmax = 10,40
        numin,numax = 900,1100
        nnu = 400 # number of frequency channels

    nside = 256
    n0 = 512 # n0^3 will be number of cells for input grid cube

    reduced_factor = 1 # Divide all dimension quanitities by this number to quickly improve speed.
    if reduced_factor!=1: nside,nnu,n0 = int(nside/reduced_factor),int(nnu/reduced_factor),int(n0/reduced_factor)

    ### Initialise cosmology for effective redshift of survey:
    import cosmo
    nu_21cm = 1420.405751#MHz
    zmin,zmax = (nu_21cm/numax) - 1,(nu_21cm/numin) - 1
    zeff = np.mean([zmin,zmax])
    cosmo.SetCosmology(builtincosmo='Planck18',z=zeff)
    Pmod = cosmo.GetModelPk(z=zeff)
    b_HI = 1.5
    OmegaHIbHI = 0.85e-3 # From MeerKATxWiggleZ constraint
    OmegaHI = OmegaHIbHI/b_HI
    import line
    T_21cm = line.T_21cm(zeff,OmegaHI)

    ### Initialise healpix environment and calculate sky survey mask and grid sizes:
    import grid
    hp0 = HEALPix(nside)
    ipix = np.arange(hp0.npix)
    ra,dec = hp0.healpix_to_lonlat(ipix)
    hpmask = np.zeros(hp0.npix)
    hpmask[ (ra.to(u.deg).value>=ramin) & (ra.to(u.deg).value<=ramax) & (dec.to(u.deg).value>=decmin) & (dec.to(u.deg).value<=decmax) ] = 1
    nu = np.linspace(numin,numax,nnu)
    ra,dec = ra[hpmask==1],dec[hpmask==1]
    ndim_0 = n0,n0,n0
    dims_0 = grid.comoving_dims(ra,dec,nu,nside,ndim_0)

    #dims_fft = grid.comoving_dims(ra,dec,nu,nside,ndim_fft)
    if dohealpy==True:
        zs = line.nu21cm_to_z(nu)
        d_c = cosmo.d_com(zs) # Comoving distance to frequency binra[ra>180] = ra[ra>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
        dims_hp = grid.get_healpy_grid_dims(ra,dec,nu,nside,d_c,dims_0)
        lx,ly,lz,nhpx,nhpy,nhpz,x0,y0,z0 = dims_hp
        npix = int(np.sum(hpmask))
        #ndim_fft = int(n0/4),int(n0/4),int(n0/4)
        ndim_fft = int(np.sqrt(npix/fft2hp_ratio)),int(np.sqrt(npix/fft2hp_ratio)),int(nnu/fft2hp_ratio)

    if dohealpy==False:
        nhpx,nhpy,nhpz = int(n0/2),int(n0/2),int(n0/2)
        #nhpx,nhpy,nhpz = int(n0/4),int(n0/4),int(n0/4)
        ndim_hp = nhpx,nhpy,nhpz
        dims_hp = grid.comoving_dims(ra,dec,nu,nside,ndim_hp)
        nnu = nhpz
        npix = nhpx*nhpy
        ndim_fft = int(nhpx/fft2hp_ratio),int(nhpy/fft2hp_ratio),int(nhpz/fft2hp_ratio)

    dims_fft = grid.comoving_dims(ra,dec,nu,nside,ndim_fft)
    lx,ly,lz,n0x,n0y,n0z = dims_0[:6]
    lx,ly,lz,nfftx,nffty,nfftz = dims_fft[:6]

    ### Determine a Gaussian beam-size to apply:
    # - defined by dish size and the beam spread at median frequency.
    if dobeam==True:
        import telescope
        D_dish = 15
        theta_FWHM,R_beam = telescope.getbeampars(D_dish,np.median(nu),verbose=True)
    else: R_beam = 0

    ### Summarise survey choices:
    print('\n - Survey summary:')
    print('Approximate survey area:', str((ramax-ramin)*(decmax-decmin)), 'deg^2')
    print('Redshift range:', str(round(zmin,3)), '< z <', str(round(zmax,3)))
    print('Central redshift:', str(round(np.median([zmin,zmax]),3)) )
    print('HealPIX pixel resolution:', str(round(hp0.pixel_resolution.to(u.deg).value,3)),'deg')
    print('Frequency channel width:', str(round((numax-numin)/nnu,3)), 'MHz' )
    ### Check dimensions for hierarchy of grids/maps:
    print('\n - Hierarchy of grid/maps:')
    print('Input grid lengths:', str([round(lx,1),round(ly,1),round(lz,1)]), '(Mpc/h)')
    print('Input grid cells (nx*ny,nz):', str([n0x*n0y,n0z]))
    print('HEALPix map (pixels,channels):', str([npix,nnu]))
    print('Output FFT grid cells (nx*ny,nz):', str([nfftx*nffty,nfftz]))
    print(' - (nx,ny,nz):',str([nfftx,nffty,nfftz]),'\n')
    #exit()

    ### Assign some k-bins for power spectra:
    import power
    nyq = np.min( [ nfftx*np.pi/lx, nffty*np.pi/ly, nfftz*np.pi/lz ] )
    kmax = 1.2*nyq
    kmin = 4*np.pi/np.max([lx,ly,lz])
    deltak = kmin/2
    kbins = np.arange(kmin,kmax,deltak)
    nkbin = len(kbins)-1

    return Pmod,b_HI,T_21cm,nside,hpmask,ramin,ramax,decmin,decmax,ra,dec,nu,dims_0,dims_hp,dims_fft,R_beam,kbins,nkbin,nyq
