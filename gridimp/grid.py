import numpy as np
import matplotlib.pyplot as plt
import pmesh
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from gridimp import model
from gridimp import cosmo
from gridimp import line
h = cosmo.H(0)/100 # use to convert astopy Mpc distances to Mpc/h
v_21cm = 1420.405751#MHz
import astropy_healpix
from astropy_healpix import HEALPix

def comoving_dims(ra,dec,nu,nside,ndim=None,W=None,frame='icrs'):
    '''Obtain lengths and origins of Cartesian comoving grid that encloses a
    sky map with (RA,Dec,nu) input coordinates for the HEALPix map voxels'''
    # ndim = tuple of pixel dimensions (nx,ny,nz) use if want added to dims arrays
    # W = binary mask - dimensions will be calculated to close fit around only filled
    #      pixels (W==1) if this is given
    hp0 = HEALPix(nside)
    ra_p,dec_p = np.copy(ra.to(u.deg).value),np.copy(dec.to(u.deg).value)
    ra_p,dec_p = np.tile(ra_p[:,np.newaxis],(1,len(nu))),np.tile(dec_p[:,np.newaxis],(1,len(nu)))
    nu_p = np.tile(nu[np.newaxis,:],(len(ra),1))
    # Cut core particles since only need edges of map to convert and obtain fitted grid:
    ra_p[ra_p>180] = ra_p[ra_p>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
    coremask = (ra_p>np.min(ra_p)) & (ra_p<np.max(ra_p)) & (ra_p>np.min(ra_p)) & (dec_p<np.max(dec_p)) & (nu_p>np.min(nu_p)) & (nu_p<np.max(nu_p))
    ra_p[ra_p<0] = ra_p[ra_p<0] + 360 # Reset negative coordinates to 359,360,1 convention
    ra_p,dec_p,nu_p = ra_p[~coremask],dec_p[~coremask],nu_p[~coremask]

    # Extend boundaries particles by 5 map pixels in all directions for a buffer
    #   since later random particles (with assignment convolution) will be kicked
    #   way beyond cell centre and can fall off grid:
    dang = hp0.pixel_resolution.to(u.deg).value
    dnu = np.mean(np.diff(nu))
    ra_p[ra_p==np.min(ra_p)] -= 5*dang
    ra_p[ra_p==np.max(ra_p)] += 5*dang
    dec_p[dec_p==np.min(dec_p)] -= 5*dang
    dec_p[dec_p==np.max(dec_p)] += 5*dang
    nu_p[nu_p==np.min(nu_p)] -= 5*dnu
    nu_p[nu_p==np.max(nu_p)] += 5*dnu

    red_p = line.nu21cm_to_z(nu_p)
    x_p,y_p,z_p = SkyCoordtoCartesian(ra_p,dec_p,red_p,ramean_arr=ra.to(u.deg).value,decmean_arr=dec.to(u.deg).value,frame=frame,doTile=False)

    x0,y0,z0 = np.min(x_p),np.min(y_p),np.min(z_p)
    lx,ly,lz = np.max(x_p)-x0, np.max(y_p)-y0, np.max(z_p)-z0
    if ndim is None: nx,ny,nz = nra,ndec,nnu
    else: nx,ny,nz = ndim
    return [lx,ly,lz,nx,ny,nz,x0,y0,z0]

def SkyCoordtoCartesian(ra_,dec_,z,ramean_arr=None,decmean_arr=None,doTile=True,LoScentre=True,frame='icrs'):
    '''Convert (RA,Dec,z) sky coordinates into Cartesian (x,y,z) comoving coordinates
    with [Mpc/h] units.
    doTile: set True (default) if input (ra,dec,z) are coordinates of map pixels of lengths ra=(nx,ny),dec=(nz,ny),z=nz)
            set False if input are test particles/galaxy coordinates already with equal length (RA,Dec,z) for every input
    LoScentre: set True (default) to align footprint with ra=dec=0 so LoS is aligned with
                one axis (x-axis by astropy default)
    ramean_arr/decmean_arr: arrays to use for mean ra/dec values. Use if want to subtract the exact same means as done for
                              another map e.g if gridding up galaxy map and want to subtract the IM mean for consistency.
    '''
    ra = np.copy(ra_);dec = np.copy(dec_) # Define new arrays so amends don't effect global coordinates
    if ramean_arr is None: ramean_arr = np.copy(ra)
    if decmean_arr is None: decmean_arr = np.copy(dec)
    if LoScentre==True: # subtract RA Dec means to align the footprint with ra=dec=0 for LoS
        ra[ra>180] = ra[ra>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
        ramean_arr[ramean_arr>180] = ramean_arr[ramean_arr>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
        ra = ra - np.mean(ramean_arr)
        ra[ra<0] = ra[ra<0] + 360 # Reset negative coordinates to 359,360,1 convention
        ramean_arr[ramean_arr<0] = ramean_arr[ramean_arr<0] + 360 # Reset negative coordinates to 359,360,1 convention
        dec = dec - np.mean(decmean_arr)
    d = cosmo.d_com(z)/h # [Mpc]
    # Build array in shape of map to assign each entry a Cartesian (x,y,z) coordinate:
    if doTile==True:
        nx,ny = np.shape(ra)
        nz = len(z)
        ra = np.repeat(ra[:, :, np.newaxis], nz, axis=2)
        dec = np.repeat(dec[:, :, np.newaxis], nz, axis=2)
        d = np.tile(d[np.newaxis,np.newaxis,:],(nx,ny,1))
    c = SkyCoord(ra*u.degree, dec*u.degree, d*u.Mpc, frame=frame)
    # Astropy does x-axis as LoS by default, so change this by assigning z=x, x=y, y=z:
    z,x,y = c.cartesian.x.value*h, c.cartesian.y.value*h, c.cartesian.z.value*h
    return x,y,z

def ParticleSampling(delta,dims_0,dims_1,Np=1,sample_ingrid=True):
    '''Use for going from Cartesian grid (dims_0) to different size Cartesian grid (dims_1)'''
    '''Create particles that lie in centre of cells and then randomly generate additional
    satellite particles kicked by random half-cell distance away from cell centre'''
    # sample_ingrid: True (default) will sample Np particles per cell of input grid
    #                False will sample Np particles per cell over output grid (with _rg index)
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims_0
    if sample_ingrid==False: lx,ly,lz,nx_rg,ny_rg,nz_rg,x0,y0,z0 = dims_1
    xbins,ybins,zbins = np.linspace(x0,x0+lx,nx+1),np.linspace(y0,y0+ly,ny+1),np.linspace(z0,z0+lz,nz+1)
    if sample_ingrid==False: xbins_rg,ybins_rg,zbins_rg = np.linspace(x0,x0+lx,nx_rg+1),np.linspace(y0,y0+ly,ny_rg+1),np.linspace(z0,z0+lz,nz_rg+1)
    # First create particles at cell centres:
    if sample_ingrid==True:
        xp0,yp0,zp0 = (xbins[1:]+xbins[:-1])/2,(ybins[1:]+ybins[:-1])/2,(zbins[1:]+zbins[:-1])/2 #centre of bins
        xp0,yp0,zp0 = np.tile(xp0[:,np.newaxis,np.newaxis],(1,ny,nz)),np.tile(yp0[np.newaxis,:,np.newaxis],(nx,1,nz)),np.tile(zp0[np.newaxis,np.newaxis,:],(nx,ny,1))
    if sample_ingrid==False:
        xp0,yp0,zp0 = (xbins_rg[1:]+xbins_rg[:-1])/2,(ybins_rg[1:]+ybins_rg[:-1])/2,(zbins_rg[1:]+zbins_rg[:-1])/2 #centre of bins
        xp0,yp0,zp0 = np.tile(xp0[:,np.newaxis,np.newaxis],(1,ny_rg,nz_rg)),np.tile(yp0[np.newaxis,:,np.newaxis],(nx_rg,1,nz_rg)),np.tile(zp0[np.newaxis,np.newaxis,:],(nx_rg,ny_rg,1))
    xp0,yp0,zp0 = np.ravel(xp0),np.ravel(yp0),np.ravel(zp0)
    if sample_ingrid==True: Hx,Hy,Hz = lx/nx,ly/ny,lz/nz
    if sample_ingrid==False: Hx,Hy,Hz = lx/nx_rg,ly/ny_rg,lz/nz_rg
    xp,yp,zp = np.array([]),np.array([]),np.array([])
    if sample_ingrid==True: loop_p = int(Np-1) # chose to include particles at voxel centre
    if sample_ingrid==False: loop_p = int(Np) # don't include particles at voxel centre, thus need an extra random
    for i in range(loop_p):
        # Satellite xp_s particles uniformly random 0<x<H/2 from cell centred particels xp0:
        xp = np.append(xp,xp0 + np.random.uniform(-Hx/2,Hx/2,np.shape(xp0)))
        yp = np.append(yp,yp0 + np.random.uniform(-Hy/2,Hy/2,np.shape(yp0)))
        zp = np.append(zp,zp0 + np.random.uniform(-Hz/2,Hz/2,np.shape(zp0)))
    if sample_ingrid==True: # append particle at voxel centre:
        xp = np.append(xp0,xp) # include cell centre particles
        yp = np.append(yp0,yp) # include cell centre particles
        zp = np.append(zp0,zp) # include cell centre particles
    ixbin = np.digitize(xp,xbins)-1
    iybin = np.digitize(yp,ybins)-1
    izbin = np.digitize(zp,zbins)-1
    cellvals = delta[ixbin,iybin,izbin] # cell values associated with each particle
    ### Filter out particles from empty pixels:
    xp = xp[cellvals!=0]
    yp = yp[cellvals!=0]
    zp = zp[cellvals!=0]
    cellvals = cellvals[cellvals!=0]
    return xp,yp,zp,cellvals

def lightcone_healpy(physmap,dims0,ra,dec,nu,nside,W=None,Np=3,frame='icrs',verbose=False):
    '''Regrid density/temp field in comoving [Mpc/h] cartesian space, into lightcone
    with (RA,Dec,z) - uses HEALPix (ra,dec) 1D array ring coordinate scheme'''
    ### Produce Np test particles per healpix map voxel for regridding:
    ra_p,dec_p,nu_p = SkyPixelParticles_healpy(ra,dec,nu,nside,map=None,W=W,Np=Np)
    red_p = line.nu21cm_to_z(nu_p)
    ### Bin particles into Cartesian bins to match each with input cell values:
    x_p,y_p,z_p = SkyCoordtoCartesian(ra_p.to(u.deg).value,dec_p.to(u.deg).value,red_p,ramean_arr=ra.to(u.deg).value,decmean_arr=dec.to(u.deg).value,frame=frame,doTile=False)
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    dx,dy,dz = lx/nx,ly/ny,lz/nz
    xbins = np.linspace(x0,x0+lx,nx+1)
    ybins = np.linspace(y0,y0+ly,ny+1)
    zbins = np.linspace(z0,z0+lz,nz+1)
    ixbin = np.digitize(x_p,xbins)-1
    iybin = np.digitize(y_p,ybins)-1
    izbin = np.digitize(z_p,zbins)-1
    cellvals = physmap[ixbin,iybin,izbin] # cell values associated with each particle
    ### Create healpy map to populate with sampling particles:
    hp0 = HEALPix(nside)
    ipix = hp0.lonlat_to_healpix(ra,dec)
    hpmap,counts = np.zeros((hp0.npix,len(nu))),np.zeros((hp0.npix,len(nu)))
    ipix_p = hp0.lonlat_to_healpix(ra_p,dec_p)
    dnu = np.mean(np.diff(nu))
    nubins = np.linspace(nu[0]-dnu/2,nu[-1]+dnu/2,len(nu)+1)
    inubin = np.digitize(nu_p,nubins)-1
    np.add.at( hpmap , (ipix_p,inubin) , cellvals )
    np.add.at( counts , (ipix_p,inubin) , 1 )
    # Average the map since multiple particle values may enter same pixel:
    hpmap[hpmap!=0] = hpmap[hpmap!=0]/counts[hpmap!=0]
    if verbose==True:
        print('\nLightcone sampling summary:')
        print(' - Minimum number of particles in map voxel: '+str(np.min(counts[ipix])))
        print(' - Maximum number of particles in map voxel: '+str(np.max(counts[ipix])))
        print(' - Mean number of particles in map voxel: '+str(np.round(np.mean(counts[ipix]),3)))
        print(' - Number of missing particles: '+str(int(len(ra_p) - np.sum(counts))))
    return hpmap

def SkyPixelParticles_healpy(ra,dec,nu,nside,map=None,W=None,Np=1):
    '''Create particles that lie in centre of ra,dec,nu voxels, then randomly generate
    additional particles kicked by random half-pixel distances away from voxel centre'''
    # Np = number of particles generated in each healpy pixel (default 1 only assigns particles at pixel centre)
    hp0 = HEALPix(nside)
    ipix = hp0.lonlat_to_healpix(ra,dec)
    dnu = np.mean(np.diff(nu))
    # Begin with particles at pixel centres:
    ra_p,dec_p = np.copy(ra),np.copy(dec)
    ra_p,dec_p = np.tile(ra_p[:,np.newaxis],(1,len(nu))),np.tile(dec_p[:,np.newaxis],(1,len(nu)))
    if Np==1: ra_p,dec_p = np.ravel(ra_p),np.ravel(dec_p)
    if Np>1: # Add satellite particles using astropy-healpix ipix offsets (dx,dy)
        for i in range(Np-1):
            for i in range(len(nu)): # Create different randoms at each channel
                dx,dy = np.random.uniform(0,1,len(ipix)),np.random.uniform(0,1,len(ipix))
                ra_p_i,dec_p_i = hp0.healpix_to_lonlat(ipix,dx,dy)
                ra_p = np.append(ra_p,ra_p_i)
                dec_p = np.append(dec_p,dec_p_i)
        del ra_p_i; del dec_p_i
    # Do similar for frequency coordinates of each particle starting with channel centres:
    nu_p = np.tile(nu[np.newaxis,:],(len(ipix),1))
    if Np==1: nu_p = np.ravel(nu_p)
    if Np>1:
        nupkick = np.array([])
        for i in range(Np-1):
            nupkick = np.append(nupkick, nu_p + np.random.uniform(-dnu/2,dnu/2,np.shape(nu_p)) )
        nu_p = np.append(nu_p,nupkick); del nupkick
    if map is not None: # Associate particles with optional input map if sampling that
        ipix_p = hp0.lonlat_to_healpix(ra_p,dec_p)
        nubins = np.linspace(nu[0]-dnu/2,nu[-1]+dnu/2,len(nu)+1)
        inubin = np.digitize(nu_p,nubins)-1
        pixvals = map[ipix_p,inubin] # cell values associated with each particle
    if W is not None:
        W_p = W[ipix_p,inubin] # use to discard particles from empty pixels
        ra_p,dec_p,nu_p = ra_p[W_p==1],dec_p[W_p==1],nu_p[W_p==1]
        if map is not None: pixvals = pixvals[W_p==1]
    if map is None: return ra_p,dec_p,nu_p
    if map is not None: return ra_p,dec_p,nu_p,pixvals

def mesh(x,y,z,T,dims,window='nnb',compensate=True,interlace=False,verbose=False):
    '''Utilises pmesh to place interpolate particles onto a grid for a given
    interpolation window function'''
    # window options are zeroth to 3rd order: ['ngp','cic','tsc','pcs']
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims
    if window=='ngp': window = 'nnb' # pmesh uses nnb for nearest grid point
    # Correct for pmesh half-cell shifting in output numpy array:
    Hx,Hy,Hz = lx/nx,ly/ny,lz/nz
    pos = np.swapaxes( np.array([x-x0-Hx/2,y-y0-Hy/2,z-z0-Hz/2]), 0,1)
    # Use pmesh to create field and paint with chosed anssigment window:
    pm0 = pmesh.pm.ParticleMesh(BoxSize=[lx,ly,lz], Nmesh=[nx,ny,nz])
    real1 = pmesh.pm.RealField(pm0)
    real1[:] = 0
    pm0.paint(pos, mass=T, resampler=window, hold=True, out=real1)
    ### Normalise by particle count entering cell:
    counts = pm0.paint(pos, resampler=window)
    real1[counts!=0] /= counts[counts!=0]
    # Create binary mask (W01), required because convolved pmesh is non-zero everywhere:
    xbins,ybins,zbins = np.linspace(x0,x0+lx,nx+1),np.linspace(y0,y0+ly,ny+1),np.linspace(z0,z0+lz,nz+1)
    W01 = np.histogramdd((x,y,z),bins=(xbins,ybins,zbins))[0]
    W01[W01!=0] = 1
    if verbose==True:
        counts = counts.preview()
        print('\nCartesian regridding summary:')
        print(' - Minimum number of particles in grid cell: '+str(np.min(counts)))
        print(' - Maximum number of particles in grid cell: '+str(np.max(counts)))
        print(' - Mean number of particles in grid cell: '+str(np.round(np.mean(counts[counts!=0]),3)))
        print(' - Number of missing particles: '+str(int(len(x) - np.sum(counts))))
    if compensate==True: # apply W(k) correction in Fourier space to field:
        c1 = real1.r2c()
        c1 /= model.W_mas(dims,window)
        if interlace==True: # Create a second shifted field (following Nbodykit:
            # https://github.com/bccp/nbodykit/blob/376c9d78204650afd9af81d148b172804432c02f/nbodykit/source/mesh/catalog.py#L11
            real2 = pmesh.pm.RealField(pm0)
            real2[:] = 0
            shifted = pm0.affine.shift(0.5)
            pm0.paint(pos, mass=T, resampler=window, transform=shifted, hold=True, out=real2)
            ### Normalise by particle count entering cell:
            counts = pm0.paint(pos, resampler=window, transform=shifted)
            real2[counts!=0] /= counts[counts!=0]
            ### Apply W(k) correction in Fourier space to field:
            c2 = real2.r2c()
            c2 /= model.W_mas(dims,window)
            # Interlace both fields (again following NBK example):
            H = [lx/nx,ly/ny,lz/nz]
            for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                kH = sum(k[i] * H[i] for i in range(3))
                s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * np.exp(0.5 * 1j * kH)
        c1.c2r(real1) # FFT back to real-space
    map = real1.preview()
    return map,W01,counts

def get_healpy_grid_dims(ra,dec,nu,nside,d_c,dims):
    '''Approximate the healpy pixel dimensions in Mpc/h for modelling'''
    # d_c: comoving distances to each frequnecy channel
    hp0 = HEALPix(nside)
    ipix = hp0.lonlat_to_healpix(ra,dec)
    dang = hp0.pixel_resolution.to(u.deg).value
    s_pix = np.mean(d_c) * np.radians(dang)
    s_para = np.mean( d_c[:-1] - d_c[1:] )
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims
    nxhp,nyhp,nzhp = round(lx/s_pix), round(ly/s_pix), round(lz/s_para)
    if nzhp % 2 != 0: # number is odd. Needs to be even
        if int(lz/s_para)!=nzhp: # originally rounded up, so round down to nearest even intg
            nzhp = int(lz/s_para)
        if int(lz/s_para)==nzhp: # originall rounded down, so round up to nearest even intg
            nzhp = round(1 + lz/s_para)
    return [lx,ly,lz,nxhp,nyhp,nzhp,x0,y0,z0]

def compress_healpix_map(map,mask,uncompress=False):
    nnu = np.shape(map)[1]
    npix = int(np.sum(mask))
    if uncompress==False:
        map_compress = np.zeros((npix,nnu))
        for i in range(nnu):
            map_compress[:,i] = map[:,i][mask==1]
        return map_compress
    if uncompress==True:
        map_compress = np.copy(map)
        map = np.zeros((len(mask),nnu))
        for i in range(nnu):
            map[:,i][mask==1] = map_compress[:,i]
        return map
