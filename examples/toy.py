import numpy as np
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix
from astropy import units as u

##### CHANGE THIS TO PIP INSTALL GRIDIMP ##############
import sys
sys.path.insert(1, '/Users/user/Documents/gridimp/gridimp')
sys.path.insert(1, '/users/scunnington/gridimp/gridimp')
sys.path.insert(1, '/Users/user/Documents/gridimp/data')
sys.path.insert(1, '/users/scunnington/gridimp/data')

ramin,ramax = 10,30
decmin,decmax = 10,30.1 #[need 30.1 otherwise differences on Ilifu and local ?????]
numin,numax = 925.5,1063.5
nnu = 60 # number of frequency channels
nside = 128
n0 = 256 # n0^3 will be number of cells for input grid cube

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
R_beam = 0 # no beam for this test

import mock
import telescope
import power
import line

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

zs = line.nu21cm_to_z(nu)
d_c = cosmo.d_com(zs) # Comoving distance to frequency binra[ra>180] = ra[ra>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
dims_hp = grid.get_healpy_grid_dims(ra,dec,nu,nside,d_c,dims_0)
lx,ly,lz,nhpx,nhpy,nhpz,x0,y0,z0 = dims_hp
npix = int(np.sum(hpmask))
ndim_fft = int(np.sqrt(npix/2)),int(np.sqrt(npix/2)),int(nnu/2)

dims_fft = grid.comoving_dims(ra,dec,nu,nside,ndim_fft)
lx,ly,lz,n0x,n0y,n0z = dims_0[:6]
lx,ly,lz,nfftx,nffty,nfftz = dims_fft[:6]

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

### Assign some k-bins for power spectra:
import power
nyq = np.min( [ nfftx*np.pi/lx, nffty*np.pi/ly, nfftz*np.pi/lz ] )
kmax = 1.2*nyq
kmin = 4*np.pi/np.max([lx,ly,lz])
deltak = kmin/2
kbins = np.arange(kmin,kmax,deltak)
nkbin = len(kbins)-1

### Generate simulated LIM survey and regridded FFT field:
f_0 = mock.Generate(Pmod,dims=dims_0,b=b_HI,Tbar=T_21cm,doRSD=False)
# Create healpy sky map or "lightcone":
map = grid.lightcone_healpy(f_0,dims_0,ra,dec,nu,nside,Np=5,verbose=True)

# Regrid to FFT:
Np = 5 # number of sampling particles per map voxel for regridding
ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles_healpy(ra,dec,nu,nside,map,Np=Np)
xp,yp,zp = grid.SkyCoordtoCartesian(ra_p.to(u.deg).value,dec_p.to(u.deg).value,line.nu21cm_to_z(nu_p),ramean_arr=ra.to(u.deg).value,decmean_arr=dec.to(u.deg).value,doTile=False)
window = 'ngp' # mass assignment function
compensate = True # correct for interpolaton effects at field level
interlace = False # interlace FFT fields
f_fft,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims_fft,window,compensate,interlace,verbose=True)
Pk,k,nmodes = power.Pk(f_fft,dims_fft,kbins,w1=W_fft,W1=W_fft)

plt.plot(k,Pk)
plt.loglog()
plt.show()
