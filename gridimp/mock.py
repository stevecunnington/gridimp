'''
Mock generation code: have taken parts from Steve Murray's Power box [https://github.com/steven-murray/powerbox/blob/master/powerbox]
however this only works for cubic boxes where nx=ny=nz - so need to use this generalised script
'''
import numpy as np
import matplotlib.pyplot as plt
try: # See if pyfftw is installed and use this if so for increased speed performance - see powerbox documentation
    from pyfftw import empty_aligned as empty
    HAVE_FFTW = True
except ImportError:
    from numpy import empty as empty
    HAVE_FFTW = False

#### Setting false by default since getting FFTW errors
HAVE_FFTW = False

from gridimp import dft # power box script

def getkspec(nx,ny,nz,dx,dy,dz,doRSD=False):
    '''Obtain 3D grid of k-modes - different to power.getkspec; this places low-k
          at centre of the array'''
    kx = dft.fftfreq(nx, d=dx, b=1)
    ky = dft.fftfreq(ny, d=dy, b=1)
    kz = dft.fftfreq(nz, d=dz, b=1)
    kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
    if doRSD==False: return kspec
    if doRSD==True:
        # Calculate mu-spectrum as needed for RSD application:
        k0mask = kspec==0
        kspec[k0mask] = 1.
        muspec = np.absolute(kz[np.newaxis,np.newaxis,:])/kspec
        kspec[k0mask] = 0.
        del k0mask
        return kspec,muspec

def _make_hermitian(mag, pha):
    #### Credit Steven Murray and Powerbox: https://github.com/steven-murray/powerbox/blob/master/powerbox
    revidx = (slice(None, None, -1),) * len(mag.shape)
    mag = (mag + mag[revidx]) / np.sqrt(2)
    pha = (pha - pha[revidx]) / 2 + np.pi
    return mag * (np.cos(pha) + 1j * np.sin(pha))

def gauss_hermitian(lx,ly,lz,nx,ny,nz):
    #### Credit Steven Murray and Powerbox: https://github.com/steven-murray/powerbox/blob/master/powerbox
    "A random array which has Gaussian magnitudes and Hermitian symmetry"
    np.random.seed(seed_)
    mag = np.random.normal(0,1,(nx+1,ny+1,nz+1))
    pha = 2*np.pi * np.random.uniform(size=(nx+1,ny+1,nz+1))
    dk = _make_hermitian(mag, pha)
    return dk[:-1,:-1,:-1] # Put back to even array by trimming off pixels

def Generate(Pmod,dims,b=1,f=0,Tbar=1,doRSD=False,LogNorm=True,seed=None,W=None):
    ### Generate a mock field
    # Default is to do a logN mock but if a Gaussian mock is required set LogNorm=False
    # seed: manually set this to same number to geneate fields to cross-correlate
    # W: optional binary survey selection function
    if seed is None: seed = np.random.randint(0,1e6)
    global seed_; seed_=seed
    lx,ly,lz,nx,ny,nz = dims[:6]
    if f!=0: doRSD=True
    # Works for even number of cells - if odd required add one and remove at end:
    x_odd,y_odd,z_odd = False,False,False # Assume even dimensions
    if nx%2!=0:
        x_odd = True
        lx = lx + lx/nx
        nx = nx + 1
    if ny%2!=0:
        y_odd = True
        ly = ly + ly/ny
        ny = ny + 1
    if nz%2!=0:
        z_odd = True
        lz = lz + lz/nz
        nz = nz + 1
    dx,dy,dz = lx/nx,ly/ny,lz/nz # Resolution
    vol = lx*ly*lz # Define volume from new grid size
    delta = empty((nx,ny,nz), dtype='complex128')
    if doRSD==False: kspec = getkspec(nx,ny,nz,dx,dy,dz)
    if doRSD==True: kspec,muspec = getkspec(nx,ny,nz,dx,dy,dz,doRSD)
    pkspec = np.zeros(np.shape(kspec))
    pkspec[kspec!=0] = 1/vol * Pmod(kspec[kspec!=0])
    if doRSD==False: pkspec = b**2 * pkspec
    if doRSD==True: pkspec = b**2 * (1 + (f/b)*muspec**2)**2 * pkspec
    if LogNorm==True:
        # Inverse Fourier transform to obtain the correlation function
        xigrid = vol * np.real(dft.ifft(pkspec, L=[lx,ly,lz], a=1, b=1)[0])
        xigrid = np.log(1 + xigrid) # Transform the correlation function
        pkspec = np.abs( dft.fft(xigrid, L=[lx,ly,lz], a=1, b=1)[0] )
        pkspec[kspec==0] = 0
    delta = np.sqrt(pkspec) * gauss_hermitian(lx,ly,lz,nx,ny,nz)
    if LogNorm==True: delta = np.sqrt(vol) * dft.ifft(delta, L=[lx,ly,lz], a=1, b=1)[0]
    if LogNorm==False: delta = vol * dft.ifft(delta, L=[lx,ly,lz], a=1, b=1)[0]
    delta = np.real(delta)
    if LogNorm==True:
        # Return log-normal density field delta_LN
        delta = np.exp(delta - np.var(delta)/2) - 1
    if x_odd==True: delta = delta[:-1,:,:]
    if y_odd==True: delta = delta[:,:-1,:]
    if z_odd==True: delta = delta[:,:,:-1]
    delta *= Tbar # multiply by Tbar if not set to 1
    if W is not None: # Assumes binary window W(0,1)
        # a more complex window can be used for galaxy mocks if Poisson sampling
        delta[W==0] = 0
    return delta
