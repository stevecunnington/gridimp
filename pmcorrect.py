import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/user/Documents/MeerKAT/meerpower/meerpower')
sys.path.insert(1, '/idia/projects/hi_im/meerpower/meerpower')
import pmesh

ncell = 256
L = 600
#L = 1080
nx,ny,nz = ncell,ncell,ncell
lx,ly,lz = L,L,L
x0,y0,z0 = [0,0,0]
dims = [lx,ly,lz,nx,ny,nz]
dims0 = [lx,ly,lz,nx,ny,nz,x0,y0,z0]
Vcell = (lx*ly*lz)/(nx*ny*nz)
ncell_rg = 128
nx_rg,ny_rg,nz_rg = ncell_rg,ncell_rg,ncell_rg
dims_rg = [lx,ly,lz,nx_rg,ny_rg,nz_rg]
dims0_rg = [lx,ly,lz,nx_rg,ny_rg,nz_rg,x0,y0,z0]

import cosmo
cosmo.SetCosmology(builtincosmo='Planck18',z=0.55,UseCLASS=True)
Pmod = cosmo.GetModelPk(z=0.55,UseCLASS=True)
import power
import mock
nkbin = 60
kmin,kmax = 0.02,0.8
kbins = np.linspace(kmin,kmax,nkbin+1)

import mock
f0_mock = mock.Generate(Pmod,dims=dims,b=1,f=0,Tbar=1,doRSD=False,seed=None,W=None)

def ParticleSampling(delta,dims0,dims0_rg,Np=1,sample_ingrid=True):
    '''Create particles that lie in centre of cells and then randomly generate additional
    satellite particles kicked by random half-cell distance away from cell centre'''
    # sample_ingrid: True (default) will sample Np particles per cell of input grid
    #                False will sample Np particles per cell over output grid
    # particleshift: set True to shift all output particles by H/2 to avoid shifting
    #                   mesh in pmesh painting output
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    #lx,ly,lz,nx_rg,ny_rg,nz_rg,x0,y0,z0 = dims0_rg
    #Hx_rg,Hy_rg,Hz_rg = lx/nx_rg,ly/ny_rg,lz/nz_rg
    xbins,ybins,zbins = np.linspace(x0,x0+lx,nx+1),np.linspace(y0,y0+ly,ny+1),np.linspace(z0,z0+lz,nz+1)
    # First create particles at cell centres:
    xp0,yp0,zp0 = (xbins[1:]+xbins[:-1])/2,(ybins[1:]+ybins[:-1])/2,(zbins[1:]+zbins[:-1])/2 #centre of bins
    xp0,yp0,zp0 = np.tile(xp0[:,np.newaxis,np.newaxis],(1,ny,nz)),np.tile(yp0[np.newaxis,:,np.newaxis],(nx,1,nz)),np.tile(zp0[np.newaxis,np.newaxis,:],(nx,ny,1))
    xp0,yp0,zp0 = np.ravel(xp0),np.ravel(yp0),np.ravel(zp0)
    Hx,Hy,Hz = lx/nx,ly/ny,lz/nz
    xp,yp,zp = np.array([]),np.array([]),np.array([])
    for i in range(Np-1):
        # Satellite xp_s particles uniformly random 0<x<H/2 from cell centred particels xp0:
        xp = np.append(xp,xp0 + np.random.uniform(-Hx/2,Hx/2,np.shape(xp0)))
        yp = np.append(yp,yp0 + np.random.uniform(-Hy/2,Hy/2,np.shape(yp0)))
        zp = np.append(zp,zp0 + np.random.uniform(-Hz/2,Hz/2,np.shape(zp0)))
    xp = np.append(xp0,xp) # include cell centre particles
    yp = np.append(yp0,yp) # include cell centre particles
    zp = np.append(zp0,zp) # include cell centre particles
    ixbin = np.digitize(xp,xbins)-1
    iybin = np.digitize(yp,ybins)-1
    izbin = np.digitize(zp,zbins)-1
    cellvals = delta[ixbin,iybin,izbin] # cell values associated with each particle
    cellvals /= (Np * (nx*ny*nz)/(nx_rg*ny_rg*nz_rg))
    return xp,yp,zp,cellvals

def assign_brightness(x,y,z,T,dims0,window='nnb',particleshift=True,interlace=False):
    lx,ly,lz,nx,ny,nz,x0,y0,z0 = dims0
    if particleshift==False: pos = np.swapaxes( np.array([x-x0,y-y0,z-z0]), 0,1)
    if particleshift==True: # Correct for pmesh half-cell shifting in output numpy array
        Hx,Hy,Hz = lx/nx,ly/ny,lz/nz
        pos = np.swapaxes( np.array([x-x0-Hx/2,y-y0-Hy/2,z-z0-Hz/2]), 0,1)
    pm0 = pmesh.pm.ParticleMesh(BoxSize=[lx,ly,lz], Nmesh=[nx,ny,nz])
    pm1 = pm0.paint(pos, mass=T, resampler=window)
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

Np = 2

### test MK mapping step:
'''
import Init
import HItools
import grid
meerpower_path = '/Users/user/Documents/MeerKAT/meerpower/'
filestem = meerpower_path+'localdata/'
map_file = filestem + 'Nscan966_Tsky_cube_p0.3d_sigma3.0_iter2.fits'
counts_file = filestem + 'Nscan966_Npix_count_cube_p0.3d_sigma3.0_iter2.fits'
MKmap,w_HI,W_HI,counts_HI,dims,ra,dec,nu,wproj = Init.ReadIn(map_file,counts_file)
nuind = 20
MKmap,w_HI,W_HI,counts_HI = MKmap[:,:,:nuind],w_HI[:,:,:nuind],W_HI[:,:,:nuind],counts_HI[:,:,:nuind]
nu = nu[:nuind]
ndim = ncell,ncell,ncell
ndim_rg = ncell_rg,ncell_rg,ncell_rg
print(dims0_rg)
dims,dims0 = grid.comoving_dims(ra,dec,nu,wproj,ndim,W=None)
dims_rg,dims0_rg = grid.comoving_dims(ra,dec,nu,wproj,ndim_rg,W=None)
#map = grid.lightcone(f0_mock,dims0,ra,dec,nu,wproj,W=None,Np=Np,verbose=True)
#ra_p,dec_p,nu_p,cellvals = grid.SkyPixelParticles(ra,dec,nu,wproj,map=map,W=None,Np=Np)
#xp,yp,zp = grid.SkyCoordtoCartesian(ra_p,dec_p,HItools.Freq2Red(nu_p),doTile=False)
'''
xp,yp,zp,cellvals = ParticleSampling(f0_mock,dims0,dims0_rg,Np,sample_ingrid=True)

f0 = assign_brightness(xp,yp,zp,cellvals * (nx*ny*nz)/(nx_rg*ny_rg*nz_rg),dims0,window='pcs',interlace=False)
Pk0,k,nmodes = power.Pk(f0,f0,dims,kbins,corrtype='HIauto',W_alias='PCS',doNGPcorrect=False,Pmod=Pmod)
import model
pkmod,k = model.PkMod(Pmod,dims,kbins,1,1,0,0,Tbar1=1,Tbar2=1,r=1,R_beam1=0,R_beam2=0,interpkbins=True,MatterRSDs=False,gridinterp=True)[0:2]
plt.axhline(1,ls='--',color='black',lw=0.8)
nyq = ncell*np.pi/L
plt.axvline(nyq,color='red',ls=':',lw=2,label=r'$k^{\rm rg}_{\rm Nyq}$')
plt.plot(k,Pk0/pkmod)
plt.show()
exit()

f1_nnb = assign_brightness(xp,yp,zp,cellvals,dims0_rg,window='nnb')
f1_cic = assign_brightness(xp,yp,zp,cellvals,dims0_rg,window='cic')
f1_cic_int = assign_brightness(xp,yp,zp,cellvals,dims0_rg,window='cic',interlace=True)
f1_tsc = assign_brightness(xp,yp,zp,cellvals,dims0_rg,window='tsc')
f1_tsc_int = assign_brightness(xp,yp,zp,cellvals,dims0_rg,window='tsc',interlace=True)
f1_pcs = assign_brightness(xp,yp,zp,cellvals,dims0_rg,window='pcs')
f1_pcs_int = assign_brightness(xp,yp,zp,cellvals,dims0_rg,window='pcs',interlace=True)
'''
f0 = assign_brightness(xp,yp,zp,cellvals * (nx*ny*nz)/(nx_rg*ny_rg*nz_rg),dims0,window='nnb',particleshift=True)
f1_nnb = assign_brightness(xp,yp,zp,cellvals,dims0_rg,window='nnb',particleshift=True)
plt.imshow(np.mean(f0,2))
plt.colorbar()
plt.figure()
plt.imshow(np.mean(f1_nnb,2))
plt.colorbar()
plt.show()
exit()
'''


Pk1_nnb,k,nmodes = power.Pk(f1_nnb,f1_nnb,dims_rg,kbins,corrtype='HIauto',W_alias='NGP',doNGPcorrect=False,Pmod=Pmod)
Pk1_cic,k,nmodes = power.Pk(f1_cic,f1_cic,dims_rg,kbins,corrtype='HIauto',W_alias='CIC',doNGPcorrect=False,Pmod=Pmod)
Pk1_cic_int,k,nmodes = power.Pk(f1_cic_int,f1_cic_int,dims_rg,kbins,corrtype='HIauto',W_alias='CIC',doNGPcorrect=False,Pmod=Pmod)
Pk1_tsc,k,nmodes = power.Pk(f1_tsc,f1_tsc,dims_rg,kbins,corrtype='HIauto',W_alias='TSC',doNGPcorrect=False,Pmod=Pmod)
Pk1_tsc_int,k,nmodes = power.Pk(f1_tsc_int,f1_tsc_int,dims_rg,kbins,corrtype='HIauto',W_alias='TSC',doNGPcorrect=False,Pmod=Pmod)
Pk1_pcs,k,nmodes = power.Pk(f1_pcs,f1_pcs,dims_rg,kbins,corrtype='HIauto',W_alias='PCS',doNGPcorrect=False,Pmod=Pmod)
Pk1_pcs_int,k,nmodes = power.Pk(f1_pcs_int,f1_pcs_int,dims_rg,kbins,corrtype='HIauto',W_alias='PCS',doNGPcorrect=False,Pmod=Pmod)

nyq = ncell*np.pi/L
nyqx_rg = ncell_rg*np.pi/L

plt.axhline(1,lw=0.8,color='black',ls=':')
plt.ylim(0.8,1.2)
plt.xlim(left=k[0],right=nyqx_rg*1.1)
plt.plot(k,Pk1_nnb/Pk0,label='No noise (NGP)',color='black')
plt.plot(k,Pk1_cic/Pk0,label='CIC',color='tab:blue')
plt.plot(k,Pk1_cic_int/Pk0,ls='--',color='tab:blue')
plt.plot(k,Pk1_tsc/Pk0,label='TCS',color='tab:orange')
plt.plot(k,Pk1_tsc_int/Pk0,ls='--',color='tab:orange')
plt.plot(k,Pk1_pcs/Pk0,label='PCS',color='tab:red')
plt.plot(k,Pk1_pcs_int/Pk0,ls='--',color='tab:red')
plt.plot([10,20],[0,0],color='gray',ls='--',label='No interlacing')
plt.plot([10,20],[0,0],color='gray',label='Interlaced')
plt.axvline(nyqx_rg,color='red',ls=':',lw=2,label=r'$k^{\rm rg}_{\rm Nyq}$')
plt.xlabel(r'$k\,[h\,{\rm Mpc^{-1}}]$')
plt.ylabel(r'$P_{\rm rg}(k)\,/\,P_0(k)$')
plt.legend(fontsize=14,ncol=2)
plt.title('%s to %s'%(ncell,ncell_rg))
plt.savefig('plots/toy_alias.png', bbox_inches='tight')
plt.show()
