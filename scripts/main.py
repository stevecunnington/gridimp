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
survey = 'FineChannel'
fft2hp_ratio = 1
Pmod,b_HI,T_21cm,nside,hpmask,ramin,ramax,decmin,decmax,ra,dec,nu,dims_0,dims_hp,dims_fft,R_beam,kbins,nkbin,nyq = params.init(survey=survey,dobeam=dobeam,dohealpy=dohealpy,fft2hp_ratio=fft2hp_ratio)
lx,ly,lz,n0x,n0y,n0z = dims_0[:6]
lx,ly,lz,nfftx,nffty,nfftz = dims_fft[:6]

import cosmo
import mock
import grid
import telescope
import power
import line

def runPkloop(Nmock,window,compensate,interlace,R_beam,dohealpy,loadMap):

    Pk_0 = np.zeros((Nmock,nkbin))
    Pk_fft = np.zeros((Nmock,nkbin))
    for i in range(Nmock):
        print(i)
        if loadMap==False:
            f_0 = mock.Generate(Pmod,dims=dims_0,b=b_HI,Tbar=T_21cm,doRSD=False)
            if R_beam!=0:
                f_0 = telescope.gaussbeam(f_0,R_beam,dims_0)
            #Pk_0[i],k,nmodes = power.Pk(f_0,dims_0,kbins)
            if dohealpy==True:
                # Create healpy sky map or "lightcone":
                map = grid.lightcone_healpy(f_0,dims_0,ra,dec,nu,nside,Np=Np,verbose=True)
                map_compress = grid.compress_healpix_map(map,hpmask)
                np.save('/idia/projects/hi_im/steve/gridimp/data/inputmaps/map_n=%s_deg=%s_Rbeam=%s_survey=%s_%s'%(n0x,round((ramax-ramin)*(decmax-decmin),0),round(R_beam,2),survey,i),map_compress)

        if dohealpy==False: # Mimic healpy lightcone simulation creation/pixelisation
            # always ngp,no compensate/interlacing - as in real healpy map-making scenario
            xp,yp,zp,cellvals = grid.ParticleSampling(f_0,dims_0,dims_hp,Np=Np,sample_ingrid=False)
            map,W01_rg,counts = grid.mesh(xp,yp,zp,cellvals,dims_hp,window='ngp',compensate=False,interlace=False)
            W_hp = np.load('data/W01_ncell=%s.npy'%nhpx)
            map[W_hp==0] = 0
            xp,yp,zp,pixvals = grid.ParticleSampling(map,dims_hp,dims_fft,Np=Np,sample_ingrid=True)

        if dohealpy==True:
            map_compress = np.load('/idia/projects/hi_im/steve/gridimp/data/inputmaps/map_n=%s_deg=%s_Rbeam=%s_survey=%s_%s.npy'%(n0x,round((ramax-ramin)*(decmax-decmin),0),round(R_beam,2),survey,i))
            map = grid.compress_healpix_map(map_compress,hpmask,uncompress=True)
            ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles_healpy(ra,dec,nu,nside,map,Np=Np)
            xp,yp,zp = grid.SkyCoordtoCartesian(ra_p.to(u.deg).value,dec_p.to(u.deg).value,line.nu21cm_to_z(nu_p),ramean_arr=ra.to(u.deg).value,decmean_arr=dec.to(u.deg).value,doTile=False)
        # Regrid to FFT:
        f_fft,W_fft,counts = grid.mesh(xp,yp,zp,pixvals,dims_fft,window,compensate,interlace,verbose=True)
        np.save('/idia/projects/hi_im/steve/gridimp/data/W01_ncell=%s'%nfftx,W_fft)
        Pk_fft[i],k,nmodes = power.Pk(f_fft,dims_fft,kbins,w1=W_fft,W1=W_fft)

    ### Save outputs:
    if dohealpy==True: np.save('/idia/projects/hi_im/steve/gridimp/data/Pks_healpy=%s_Rbeam=%s_hp2fft=%s_survey=%s_%s_interlace=%s_compensate=%s'%(dohealpy,np.round(R_beam,2),fft2hp_ratio,survey,window,interlace,compensate),[Pk_0,Pk_fft])

loadMap = False

Nmock = 100
Np = 5

survey = 'FineChannel'
dobeam = True
fft2hp_ratio = 1
Pmod,b_HI,T_21cm,nside,hpmask,ramin,ramax,decmin,decmax,ra,dec,nu,dims_0,dims_hp,dims_fft,R_beam,kbins,nkbin,nyq = params.init(survey=survey,dobeam=dobeam,dohealpy=dohealpy,fft2hp_ratio=fft2hp_ratio)

compensate = True

#windows = ['cic','tsc','ngp','pcs']
#interlaces = [False,True]
windows = ['tsc']
interlaces = ['True']
for window in windows:
    for interlace in interlaces:
        runPkloop(Nmock,window,compensate,interlace,R_beam,dohealpy,loadMap)
exit()

survey = 'Initial'
Pmod,b_HI,T_21cm,nside,hpmask,ramin,ramax,decmin,decmax,ra,dec,nu,dims_0,dims_hp,dims_fft,R_beam,kbins,nkbin,nyq = params.init(survey=survey,dobeam=dobeam,dohealpy=dohealpy,fft2hp_ratio=fft2hp_ratio)
for compensate in compensates:
    runPkloop(Nmock,window,compensate,interlace,R_beam,dohealpy,loadMap)
exit()


'''
## No treatment biased case:
window = 'ngp'
interlace = False
compensate = False
runPkloop(Nmock,window,compensate,interlace,R_beam,dohealpy,loadMap)
exit()
'''
#'''
compensate = True
window = 'tcs'
interlace = False
loadMap = False
runPkloop(Nmock,window,compensate,interlace,R_beam,dohealpy,loadMap)
exit()
#'''

## Loop over all MAS possibilities::
loadMap = False
windows = ['ngp','pcs','cic','tsc']
interlaces = [False,True]
compensate = True
for window in windows:
    for interlace in interlaces:
        runPkloop(Nmock,window,compensate,interlace,R_beam,dohealpy,loadMap)
#'''
exit()

window = 'ngp'
interlace = False
compensate = False

Pk_0,Pk_fft = np.load('data/Pks_healpy=%s_Rbeam=%s_%s_interlace=%s_compensate=%s.npy'%(dohealpy,np.round(R_beam,2),window,interlace,compensate))

### Modelling:
import model
#pkmod_0,k,nmodes = model.PkMod(Pmod,dims_0,kbins,b1=b_HI,Tbar1=T_21cm,R_beam1=R_beam)

if dohealpy==False:
    s_para = lz/nhpz
    s_pix = np.mean([lx/nhpx,ly/nhpy])
    pkmod_hp,k,nmodes = model.PkMod(Pmod,dims_hp,kbins,b1=b_HI,Tbar1=T_21cm,R_beam1=R_beam,w1=W_hp,W1=W_hp,s_pix=s_pix,s_para=s_para,nsum=1,window='ngp')
if dohealpy==True:
    zs = line.nu21cm_to_z(nu)
    d_c = cosmo.d_com(zs) # Comoving distance to frequency binra[ra>180] = ra[ra>180] - 360 # Make continuous RA i.e. 359,360,1 -> -1,0,1 so mean RA is correct
    s_para = np.mean( cosmo.d_com(zs[:-1]) - cosmo.d_com(zs[1:]) )
    hppixwin = model.HealpixPixelWindow(nside,np.mean(d_c))
    # Resample binary window mask into the approximated healpix dimensional space:
    W_fft = np.load('data/W01_ncell=%s.npy'%nfftx)
    xp,yp,zp,cellvals = grid.ParticleSampling(W_fft,dims_fft,dims_hp,Np=1,sample_ingrid=False)
    W_hp = grid.mesh(xp,yp,zp,cellvals,dims_hp,window='ngp',compensate=False,interlace=False)[1]
    #s_para,hppixwin = None,None
    nsum = 1
    pkmod_hp,k,nmodes = model.PkMod(Pmod,dims_hp,kbins,b1=b_HI,Tbar1=T_21cm,R_beam1=R_beam,w1=W_hp,W1=W_hp,s_para=s_para,hppixwin=hppixwin,nsum=nsum,window='ngp')

plt.axhline(1,color='black',lw=0.8,ls=':')
plt.axvline(nyq,color='red',lw=1,ls='--')
plt.axvline(nyq/2,color='red',lw=1,ls='--')
#plt.errorbar(k,np.mean(Pk_0,0)/pkmod_0,np.std(Pk_0,0)/pkmod_0)
plt.errorbar(k,np.mean(Pk_fft,0)/pkmod_hp,np.std(Pk_fft,0)/pkmod_hp)
plt.show()
