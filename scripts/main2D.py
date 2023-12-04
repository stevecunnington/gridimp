import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from astropy import units as u

##### CHANGE THIS TO PIP INSTALL GRIDIMP ##############
import sys
sys.path.insert(1, '/Users/user/Documents/gridimp/gridimp')
sys.path.insert(1, '/users/scunnington/gridimp/gridimp')
sys.path.insert(1, '/Users/user/Documents/gridimp/data')
sys.path.insert(1, '/users/scunnington/gridimp/data')

import params
dohealpy = True
dobeam = True
fft2hp_ratio = 2

### Always use FineChannel to define kperp/kpara bins for easy comparison:
survey = 'FineChannel'
Pmod,b_HI,T_21cm,nside,hpmask,ramin,ramax,decmin,decmax,ra,dec,nu,dims_0,dims_hp,dims_fft,R_beam,kbins,nkbin,nyq = params.init(survey=survey,dobeam=dobeam,dohealpy=dohealpy,fft2hp_ratio=fft2hp_ratio)
lx,ly,lz,n0x,n0y,n0z = dims_0[:6]
lx,ly,lz,nfftx,nffty,nfftz = dims_fft[:6]
nyqx,nyqy,nyqz = nfftx*np.pi/lx, nffty*np.pi/ly, nfftz*np.pi/lz
nyq_perp = np.sqrt(nyqx**2 + nyqy**2)
nyq_para = nyqz
kmin = 4*np.pi/np.max([lx,ly,lz])
kperpbins = np.linspace(kmin,nyq_perp,int(nfftx/3))
kparabins = np.linspace(kmin,nyq_para,int(nfftz/3))

### Chose version to use for results:
survey = 'Initial'
#survey = 'FineChannel'
Pmod,b_HI,T_21cm,nside,hpmask,ramin,ramax,decmin,decmax,ra,dec,nu,dims_0,dims_hp,dims_fft,R_beam,kbins,nkbin,nyq = params.init(survey=survey,dobeam=dobeam,dohealpy=dohealpy,fft2hp_ratio=fft2hp_ratio)
lx,ly,lz,n0x,n0y,n0z = dims_0[:6]
lx,ly,lz,nfftx,nffty,nfftz = dims_fft[:6]

from gridimp import cosmo
from gridimp import mock
from gridimp import grid
from gridimp import telescope
from gridimp import power
from gridimp import line
from gridimp import model

def runPkloop(Nmock,window,compensate,interlace,R_beam,dohealpy,loadMap):

    Pk_fft = np.zeros((Nmock,len(kparabins)-1,len(kperpbins)-1))
    for i in range(Nmock):
        print(i)
        if loadMap==False:
            f_0 = mock.Generate(Pmod,dims=dims_0,b=b_HI,Tbar=T_21cm,doRSD=False)
            if R_beam!=0:
                f_0 = telescope.gaussbeam(f_0,R_beam,dims_0)
            # Create healpy sky map or "lightcone":
            map = grid.lightcone_healpy(f_0,dims_0,ra,dec,nu,nside,Np=Np,verbose=True)
        ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles_healpy(ra,dec,nu,nside,map,Np=Np)
        xp,yp,zp = grid.SkyCoordtoCartesian(ra_p.to(u.deg).value,dec_p.to(u.deg).value,line.nu21cm_to_z(nu_p),ramean_arr=ra.to(u.deg).value,decmean_arr=dec.to(u.deg).value,doTile=False)
        # Regrid to FFT:
        f_fft,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims_fft,window,compensate,interlace,verbose=True)
        np.save('/idia/projects/hi_im/steve/gridimp/data/W01_ncell=%s'%nfftx,W_fft)
        #np.save('data/W01_ncell=%s'%nfftx,W_fft)
        Pk_fft[i],k,nmodes = power.Pk2D(f_fft,dims_fft,kperpbins,kparabins,w1=W_fft,W1=W_fft)

    ### Save outputs:
    np.save('/idia/projects/hi_im/steve/gridimp/data/Pks2D_healpy=%s_Rbeam=%s_hp2fft=%s_survey=%s_%s_interlace=%s_compensate=%s'%(dohealpy,np.round(R_beam,2),fft2hp_ratio,survey,window,interlace,compensate),[k,Pk_fft])
    #np.save('data/Pks2D_healpy=%s_Rbeam=%s_hp2fft=%s_survey=%s_%s_interlace=%s_compensate=%s'%(dohealpy,np.round(R_beam,2),fft2hp_ratio,survey,window,interlace,compensate),[k,Pk_0,Pk_fft])
    return k,Pk_fft

loadMap = False

Nmock = 100
Np = 5

## No treatment biased case:
window = 'ngp'
interlace = False
compensate = True
k2D,Pk2D_fft = runPkloop(Nmock,window,compensate,interlace,R_beam,dohealpy,loadMap)
exit()
k2D,Pk2D_fft = np.load('data/Pks2D_healpy=%s_Rbeam=%s_hp2fft=%s_survey=%s_%s_interlace=%s_compensate=%s.npy'%(dohealpy,np.round(R_beam,2),fft2hp_ratio,survey,window,interlace,compensate),allow_pickle=True)
#k2D,Pk2D_0,Pk2D_fft = np.load('/idia/projects/hi_im/steve/gridimp/data/Pks2D_healpy=%s_Rbeam=%s_hp2fft=%s_survey=%s_%s_interlace=%s_compensate=%s.npy'%(dohealpy,np.round(R_beam,2),fft2hp_ratio,survey,window,interlace,compensate),allow_pickle=True)

Pk2D_0 = np.mean(Pk2D_0,0)
Pk2D_fft = np.mean(Pk2D_fft,0)

nsum = 0
s_para,hppixwin = None,None
W_fft = np.load('data/W01_ncell=%s.npy'%nfftx)
#W_fft = np.load('/idia/projects/hi_im/steve/gridimp/data/W01_ncell=%s.npy'%nfftx)

xp,yp,zp,cellvals = grid.ParticleSampling(W_fft,dims_fft,dims_hp,Np=1,sample_ingrid=False)
W_hp = grid.mesh(xp,yp,zp,cellvals,dims_hp,window='ngp',compensate=False,interlace=False)[1]

#Pk2D_0,k2d,nmodes = model.PkMod(Pmod,dims_fft,kbins,b1=b_HI,Tbar1=T_21cm,R_beam1=R_beam,w1=W_fft,W1=W_fft,s_para=s_para,hppixwin=hppixwin,nsum=nsum,window='ngp',Pk2D=True,kperpbins=kperpbins,kparabins=kparabins)
Pk2D_0,k2d,nmodes = model.PkMod(Pmod,dims_hp,kbins,b1=b_HI,Tbar1=T_21cm,R_beam1=R_beam,w1=W_hp,W1=W_hp,s_para=s_para,hppixwin=hppixwin,nsum=nsum,window='ngp',Pk2D=True,kperpbins=kperpbins,kparabins=kparabins)

metric = Pk2D_fft/Pk2D_0

maxx = np.max(np.abs(metric))
maxx = 1
vmin,vmax = -maxx,maxx
vmin,vmax = 0,1.4
plt.pcolormesh(kperpbins,kparabins,metric,cmap='seismic_r',vmin=vmin,vmax=vmax)

plt.xlim(left=kperpbins[0],right=nyq_perp*0.9)
plt.ylim(bottom=kparabins[0],top=nyq_para*0.9)

k0 = nyq  # characteristic scale at which P_HI~100
kperp = np.linspace(np.min(kperpbins),k0,100)
kpara = np.linspace(0,k0,100)
k0cont = np.sqrt(k0**2 - kpara**2)
plt.plot(kperp,k0cont,ls='--',lw=3,color='red')

plt.colorbar()
plt.xlabel(r'$k_\perp [h\,{\rm Mpc}^{-1}]$')
plt.ylabel(r'$k_\parallel [h\,{\rm Mpc}^{-1}]$')
plt.show()
