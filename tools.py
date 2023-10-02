import numpy as np
import matplotlib.pyplot as plt
#'''
import scipy.ndimage
import scipy as sp
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
import astropy.wcs
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.coordinates as ac
#'''
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
import HItools
h = cosmo.H(0).value/100 # use to convert astopy Mpc distances to Mpc/h
v_21cm = 1420.405751#MHz

import grid
import pmesh

def cartesian(map,ra,dec,nu,wproj,W=None,ndim=None,Np=3,window='nnb',interlace=False,particleshift=True,frame='icrs',verbose=False):
    '''regrid (RA,Dec,z) map into comoving Cartesian coordinates (Lx,Ly,Lz [Mpc/h])'''
    ### Produce Np test particles per map voxel for regridding:
    ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=map,W=W,Np=Np)
    xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),frame=frame,doTile=False)
    # Don't use particle mins/maxs to obtain dims, instead use the generic function
    #   below to ensure consistent dimensions throughout code rather than being
    #   random particle dependent:
    dims,dims0 = grid.comoving_dims(ra,dec,nu,wproj,ndim=ndim,W=W,frame=frame)
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    # Create position array - requires axes swap: [[x..],[y..],[z..]] -> [[x,y,z],[x,y,z],...]
    if particleshift==False: pos = np.swapaxes( np.array([xp-x0,yp-y0,zp-z0]), 0,1)
    if particleshift==True: # Correct for pmesh half-cell shifting in output numpy array
        Hx,Hy,Hz = lx/nx,ly/ny,lz/nz
        pos = np.swapaxes( np.array([xp-x0-Hx/2,yp-y0-Hy/2,zp-z0-Hz/2]), 0,1)
    pm0 = pmesh.pm.ParticleMesh(BoxSize=[lx,ly,lz], Nmesh=[nx,ny,nz])
    pm1 = pm0.paint(pos, mass=pixvals, resampler=window)
    ### Normalise by particle count entering cell
    pm1_counts = pm0.paint(pos, resampler=window)
    pm1[pm1_counts!=0] /= pm1_counts[pm1_counts!=0]
    if verbose==True:
        pm1_counts = pm1_counts.preview()
        print('\nCartesian regridding summary:')
        print(' - Minimum number of particles in grid cell: '+str(np.min(pm1_counts)))
        print(' - Maximum number of particles in grid cell: '+str(np.max(pm1_counts)))
        print(' - Mean number of particles in grid cell: '+str(np.round(np.mean(pm1_counts),3)))
        print(' - Number of missing particles: '+str(int(len(xp) - np.sum(pm1_counts))))
    del pm1_counts
    if interlace==False: return pm1.preview()
    if interlace==True:
        ### from NBK: https://github.com/bccp/nbodykit/blob/376c9d78204650afd9af81d148b172804432c02f/nbodykit/source/mesh/catalog.py#L11
        real1 = pmesh.pm.RealField(pm0)
        real1[:] = 0
        # the second, shifted mesh (always needed)
        real2 = pmesh.pm.RealField(pm0)
        real2[:] = 0
        shifted = pm0.affine.shift(0.5)
        # paint to two shifted meshes
        pm0.paint(pos, mass=T, resampler=window, hold=True, out=real1)
        pm0.paint(pos, mass=T, resampler=window, transform=shifted, hold=True, out=real2)
        # compose the two interlaced fields into the final result.
        c1 = real1.r2c()
        c2 = real2.r2c()
        # and then combine
        H = [lx/nx,ly/ny,lz/nz]
        for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
            kH = sum(k[i] * H[i] for i in range(3))
            s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * np.exp(0.5 * 1j * kH)
        # FFT back to real-space
        c1.c2r(real1)
        return real1.preview()
