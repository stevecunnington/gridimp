import numpy as np
import cosmo
H0 = cosmo.H(0)
import power
from scipy.interpolate import interp1d

def PkMod(Pmod,dims,kbins=None,b1=1,b2=None,f=0,sig_v=0,Tbar1=1,Tbar2=None,r=1,R_beam1=0,R_beam2=None,sig_N=0,w1=None,w2=None,W1=None,W2=None,s_pix=None,s_para=None,hppixwin=None,nsum=0,window=None,interpkbins=True,Pk2D=False,kperpbins=None,kparabins=None):
    '''Compute model spherically averaged power spectrum P(|\vec{k}|)'''
    # Supply differnt second parameters (b2,Tbar2..etc.) if doing a cross-correlation
    #  between different tracers. If none given, then do auto correlation assuming b1==b2 etc:
    if b2 is None: b2 = b1
    if Tbar2 is None: Tbar2 = Tbar1
    if R_beam2 is None: R_beam2 = R_beam1
    if w2 is None: w2 = w1
    if W2 is None: W2 = W1
    # interpkbins = True (default), interpolates power spectrum over same grid as
    #   data, providing slower calcualtion but more accurate results
    if interpkbins==True: # If True, interpolate model Pk over same grid and bin using same pipeline as data
        if nsum==0: # no summation over aliased modes
            kspec,muspec,indep = power.getkspec(dims,FullPk=True)
            pkspecmod = PkModSpec(Pmod,dims,kspec,muspec,b1,b2,f,sig_v,Tbar1,Tbar2,r,R_beam1,R_beam2,sig_N,w1,w2,W1,W2,s_pix,s_para,hppixwin)
        else:
            if window is None:
                print('\n ERROR!: Interpolation window must be specified for aliased mode summing\n')
                exit()
            pkspecmod = PkModSpec_summed(Pmod,dims,b1,b2,f,sig_v,Tbar1,Tbar2,r,R_beam1,R_beam2,sig_N,w1,w2,W1,W2,s_pix,s_para,hppixwin,n=nsum,window=window)
        if Pk2D==False:
            pkmod,k,nmodes = power.binpk(pkspecmod,dims[:6],kbins,FullPk=True,doindep=False)
            return pkmod,k,nmodes
        if Pk2D==True:
            pk2d,k2d,nmodes = power.binpk2D(pkspecmod,dims[:6],kperpbins,kparabins,FullPk=True,doindep=False)
            return pk2d,k2d,nmodes
    if interpkbins==False: # If False, run a more approximate model build, using integration over analytical function
        kmod = (kbins[1:] + kbins[:-1]) / 2 #centre of k-bins
        beta1,beta2 = f/b1,f/b2
        deltak = [kbins[i]-kbins[i-1] for i in range(1,len(kbins))]
        if sig_N!=0: P_N = sig_N**2 * (lx*ly*lz)/(nx*ny*nz) # noise term
        else: P_N = 0
        Pk_int = lambda mu: Tbar1*Tbar2 * b1*b2*( r + (beta1 + beta2)*mu**2 + beta1*beta2*mu**4 ) / (1 + (k_i*mu*sig_v/H0)**2) * Pmod(k_i) * B_beam(mu,k_i,R_beam1) * B_beam(mu,k_i,R_beam2) + P_N
        pkmod = np.zeros(len(kmod))
        nmodes = np.zeros(len(kmod))
        for i in range(len(kmod)):
            k_i = kmod[i]
            pkmod[i] = scipy.integrate.quad(Pk_int, 0, 1)[0]
            nmodes[i] = 1 / (2*np.pi)**3 * (lx*ly*lz) * (4*np.pi*k_i**2*deltak[i]) # Based on eq14 in https://arxiv.org/pdf/1509.03286.pdf
        return pkmod,kmod,nmodes

def PkModSpec(Pmod,dims,kspec,muspec,b1,b2=None,f=0,sig_v=0,Tbar1=1,Tbar2=None,r=1,R_beam1=0,R_beam2=None,sig_N=0,w1=None,w2=None,W1=None,W2=None,s_pix=None,s_para=None,hppixwin=None):
    '''Model power in full 3D spectrum format interpolated over input grid'''
    # Supply differnt second parameters (b2,Tbar2..etc.) if doing a cross-correlation
    #  between different tracers. If none given, then do auto correlation assuming b1==b2 etc:
    if b2 is None: b2 = b1
    if Tbar2 is None: Tbar2 = Tbar1
    if R_beam2 is None: R_beam2 = R_beam1
    if w2 is None: w2 = w1
    if W2 is None: W2 = W1
    lx,ly,lz,nx,ny,nz = dims[:6]
    kspec[kspec==0] = 1 # avoid Pmod-model interpolation for k=0
    # Collect damping terms from beam/FG/channels/heapy pixelisation:
    Damp = B_beam(muspec,kspec,R_beam1) * B_beam(muspec,kspec,R_beam2) * B_pix(muspec,kspec,s_pix)**2 * B_channel(muspec,kspec,s_para)**2 * B_hppix(muspec,kspec,hppixwin)**2
    if sig_N!=0: P_N = sig_N**2 * (lx*ly*lz)/(nx*ny*nz) # noise term
    else: P_N = 0
    beta1,beta2 = f/b1,f/b2 # Include bias in Kaiser term (sensitive in quadrupole)
    pkspecmod = Damp * Tbar1*Tbar2 * b1*b2*( r + (beta1 + beta2)*muspec**2 + beta1*beta2*muspec**4 ) / (1 + (kspec*muspec*sig_v/H0)**2) * Pmod(kspec) + P_N
    if w1 is not None or w2 is not None or W1 is not None or W2 is not None: # Convolve with window
        pkspecmod = power.getpkconv(pkspecmod,dims,w1,w2,W1,W2)
    return pkspecmod

def PkModSpec_summed(Pmod,dims,b1,b2=None,f=0,sig_v=0,Tbar1=1,Tbar2=None,r=1,R_beam1=0,R_beam2=None,sig_N=0,w1=None,w2=None,W1=None,W2=None,s_pix=None,s_para=None,hppixwin=None,n=1,window='ngp'):
    '''Sum 3D model power spectra over modes displaced by 2n of nyquist frequency'''
    # n=1 (default): number of integer modes to displace by in all directions.
    # \vec{n} is (1+2n)^3 array of integers in all direction.
    # Larger n sums over more modes thus more accurate. Diminishing returns above [-1,0,1]
    if window=='ngp' or window=='nnb': p = 1
    if window=='cic': p = 2
    if window=='tsc': p = 3
    if window=='pcs': p = 4
    shiftarray = np.linspace(-n,n,(1+2*n)) # integer vectors by which to nudge the nyquist freq.
    # Supply differnt second parameters (b2,Tbar2..etc.) if doing a cross-correlation
    #  between different tracers. If none given, then do auto correlation assuming b1==b2 etc:
    if b2 is None: b2 = b1
    if Tbar2 is None: Tbar2 = Tbar1
    if R_beam2 is None: R_beam2 = R_beam1
    if w2 is None: w2 = w1
    if W2 is None: W2 = W1
    # Nyquist frequencies and k-arrays:
    lx,ly,lz,nx,ny,nz = dims[:6]
    nyqx,nyqy,nyqz = nx*np.pi/lx,ny*np.pi/ly,nz*np.pi/lz
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)[:,np.newaxis,np.newaxis]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)[np.newaxis,:,np.newaxis]
    kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[np.newaxis,np.newaxis,:]
    # Sum over integers \vec{n}=[-1,0,1] if nsum=1 as in https://arxiv.org/pdf/1902.07439.pdf step(v) pg 13:
    sum = 0
    for ix in shiftarray:
        for iy in shiftarray:
            for iz in shiftarray:
                kx_ = kx + 2*nyqx*ix
                ky_ = ky + 2*nyqy*iy
                kz_ = kz + 2*nyqz*iz
                kspec_ = np.sqrt(kx_**2 + ky_**2 + kz_**2)
                kspec_[0,0,0] = 1 # to avoid divide by zero error
                muspec_ = kz_/kspec_
                muspec_[0,0,0] = 1 # divide by k=0, means mu->1
                kspec_[0,0,0] = 0 # reset
                pkspecmod = PkModSpec(Pmod,dims,kspec_,muspec_,b1,b2,f,sig_v,Tbar1,Tbar2,r,R_beam1,R_beam2,sig_N,w1,w2,W1,W2,s_pix=s_pix,s_para=s_para,hppixwin=hppixwin)
                s_pixx = lx/nx
                s_pixy = ly/ny
                s_pixz = lz/nz
                wx = B_sinc(kx_,s_pix=s_pixx)
                wy = B_sinc(ky_,s_pix=s_pixy)
                wz = B_sinc(kz_,s_pix=s_pixz)
                W = (wx*wy*wz)**p
                sum += W**2*pkspecmod
    return sum

def B_beam(mu,k,R_beam):
    '''Gaussian smoothing from a beam with R_beam = D_com(z)*sigma_beam'''
    # Supply R_beam in [Mpc/h] units
    if R_beam==0: return 1
    return np.exp( -(1-mu**2)*k**2*R_beam**2/2 )

def B_sinc(ki,s_pix=None):
    '''Generic damping along ki direction from top-hat pixelisation. Returns
    FFT of top-hat (sinc function) kernel, defined by pixel size s_pix'''
    # ki: any anisotropic 3D karray e.g. kx,ky(or combined k_perp),kz(==k_para)
    # s_pix: pixel width in [Mpc/h]
    if s_pix is None: return 1
    q = ki*s_pix/2
    return np.divide(np.sin(q),q,out=np.ones_like(q),where=q!=0.)

def B_pix(mu,k,s_pix=None):
    '''Damping due to perpendicular binning in pixels'''
    # s_pix: pixel width in [Mpc/h]
    if s_pix is None: return 1
    k_perp = k*np.sqrt(1-mu**2)
    q = k_perp*s_pix/2
    return np.divide(np.sin(q),q,out=np.ones_like(q),where=q!=0.)

def B_channel(mu,k,s_para=None):
    '''Damping due to radial binning in redshift or frequency channels'''
    # s_para: frequency channel width in [Mpc/h]
    if s_para is None: return 1
    k_para = k*mu
    q = k_para*s_para/2
    return np.divide(np.sin(q),q,out=np.ones_like(q),where=q!=0.)

def B_hppix(mu,k,hppixwin=None):
    '''Damping from HEALPix pixelisation. hppixwin should be continuous window
    function defined using healpy routines'''
    # Assumes flat-sky and pixelisation only applied to k_perp modes
    if hppixwin is None: return 1
    k_perp = k*np.sqrt(1-mu**2)
    return hppixwin(k_perp)

def HealpixPixelWindow(nside,d_c,kperpmax=10):
    '''Use Healpy to get healpix window function:
    https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.pixwin.html
    Extend on this by extrapolating to very small scales (~linear extrapolation)
    so it can be used to damp model at very high-k where window function -> 0'''
    import healpy as hp
    lmax = 2000 # highest healpy window function calculates to
    win = hp.pixwin(nside,lmax=lmax)
    l = np.arange(len(win))
    kperp = l/d_c
    # Use n=1 polynomial (linear) fit for extrapolation of window function to high kperp
    rangefrac = 0.7 # portion of scales above to extrapolate from
                    #   - set high-ish so its a linear extrapolation to small scales
    smallscalemask = kperp > rangefrac*np.max(kperp)
    pixwin,coef = FitPolynomial(kperp[smallscalemask],win[smallscalemask],n=1,returncoef=True)
    kperp_extrap = np.linspace(np.max(kperp),kperpmax,500)
    pixwin_extrap = np.zeros(len(kperp_extrap)) # fitted function
    for i in range(2):
        pixwin_extrap += coef[-(i+1)]*kperp_extrap**i
    pixwin_extrap[pixwin_extrap<0] = 0 # set window function to zero at high kperp
    kperp = np.append(kperp,kperp_extrap)
    pixwin = np.append(win,pixwin_extrap)
    return interp1d(kperp, pixwin)

def W_mas(dims,window='nnb',FullPk=False):
    '''Hockney Eastwood mass assignment corrections'''
    if window=='nnb' or 'ngp': p = 1
    if window=='cic': p = 2
    if window=='tsc': p = 3
    if window=='pcs': p = 4
    lx,ly,lz,nx,ny,nz = dims[:6]
    nyqx,nyqy,nyqz = nx*np.pi/lx,ny*np.pi/ly,nz*np.pi/lz
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)[:,np.newaxis,np.newaxis]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)[np.newaxis,:,np.newaxis]
    if FullPk==False: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1][np.newaxis,np.newaxis,:]
    if FullPk==True: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[np.newaxis,np.newaxis,:]
    qx,qy,qz = (np.pi*kx)/(2*nyqx),(np.pi*ky)/(2*nyqy),(np.pi*kz)/(2*nyqz)
    wx = np.divide(np.sin(qx),qx,out=np.ones_like(qx),where=qx!=0.)
    wy = np.divide(np.sin(qy),qy,out=np.ones_like(qy),where=qy!=0.)
    wz = np.divide(np.sin(qz),qz,out=np.ones_like(qz),where=qz!=0.)
    return (wx*wy*wz)**p

def FitPolynomial(x,y,n,returncoef=False):
    ### Fit a polynomial of order n to a generic 1D data array [x,y]
    coef = np.polyfit(x,y,n)
    func = np.zeros(len(x)) # fitted function
    for i in range(n+1):
        func += coef[-(i+1)]*x**i
    if returncoef==False: return func
    if returncoef==True: return func,coef
