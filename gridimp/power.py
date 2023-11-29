#### Steve Cunnington code with contributions from Chris Blake's code package
# available from links in: https://arxiv.org/pdf/1902.07439.pdf

import numpy as np

def Pk(f1,dims,kbins,f2=None,corrtype='HIauto',kcuts=None,w1=None,w2=None,W1=None,W2=None):
    '''Estimate the 3D power spectrum of input fluctuation field(s)'''
    ### *** IF CROSS-CORRELATING: assumes f1 = HI field and f2 = galaxy field ****
    if f2 is None: f2,w2,W2 = f1,w1,W2 # auto-correlate if only one field given
    ######################################################################
    # f1/f2: fields to correlate (f=dT_HI for IM, f=n_g for galaxies)
    # corrtype: type of correlation to compute, options are:
    #   - corrtype='HIauto': (default) for HI auto-correlation of temp fluctuation field dT_HI = T_HI - <T_HI>
    #   - corrtype='Galauto': for galaxy auto-correlation of number counts field n_g
    #   - corrtype='Cross': for HI-galaxy cross-correlation <dT_HI,n_g>
    #   - for HI IM:   norm = None (assuming fluctuation field is the input f field)
    #   - for galxies: norm = N_g (total galaxy count in input f field)
    # w1/w2: optional field weights
    # W1/W2: optional survey selection functions
    # kcuts = [kperpmin,kparamin,kperpmax,kparamax]: If given, power spectrum only measured within this scale range
    ######################################################################
    pkspec = getpkspec(f1,f2,dims,corrtype,w1,w2,W1,W2)
    Pk,k,nmodes = binpk(pkspec,dims,kbins,kcuts)
    return Pk,k,nmodes

def Pk2D(f1,dims,kperpbins,kparabins,f2=None,corrtype='HIauto',w1=None,w2=None,W1=None,W2=None):
    '''
    Same as 1D Pk() function but for 2D cylindrical power spectrum
    '''
    if f2 is None: f2,w2,W2 = f1,w1,W2 # auto-correlate if only one field given
    pkspec = getpkspec(f1,f2,dims,corrtype,w1,w2,W1,W2)
    Pk2d,k2d,nmodes = binpk2D(pkspec,dims,kperpbins,kparabins)
    return Pk2d,k2d,nmodes

def getpkspec(f1,f2,dims,corrtype='HIauto',w1=None,w2=None,W1=None,W2=None,Ngal=None):
    '''Obtain full 3D unbinned power spectrum - follows formalism in Blake+10[https://arxiv.org/pdf/1003.5721.pdf] (sec 3.1)
     - see Pk function for variable definitions'''
    if corrtype=='Galauto' or corrtype=='Cross': Ngal = np.sum(f2) # assumes f2 is galaxy field
    lx,ly,lz,nx,ny,nz = dims[:6]
    Vcell = (lx*ly*lz) / (nx*ny*nz)
    # Apply default unity weights/windows if none given:
    if w1 is None: w1 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if w2 is None: w2 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if W1 is None: W1 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if W2 is None: W2 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if corrtype=='Galauto': W1,W2 = W1/np.sum(W1),W2/np.sum(W2) # Normalise galaxy window functions so sum(W)=1
    if corrtype=='Cross': W2 = W2/np.sum(W2) # Normalise galaxy window functions so sum(W)=1
    if corrtype=='HIauto':
        S = 0 ### DEVELOP THIS ### for thermal noise HI IM subtraction
        F1k = np.fft.rfftn(w1*f1)
        F2k = np.fft.rfftn(w2*f2)
        pkspec = np.real( F1k * np.conj(F2k) )
        return Vcell / np.sum(w1*w2) * (pkspec-S) # Normalisation with windows is NOT needed
    if corrtype=='Galauto':
        S = Ngal * np.sum(w1**2*W1) # shot-noise term
        Wk1 = np.fft.rfftn(w1*W1)
        F1k = np.fft.rfftn(w1*f1) - Ngal*Wk1
        Wk2 = np.fft.rfftn(w2*W2)
        F2k = np.fft.rfftn(w2*f2) - Ngal*Wk2
        pkspec = np.real( F1k * np.conj(F2k) )
        return Vcell / np.sum(w1*w2*W1*W2) * (pkspec-S) * 1/(Ngal**2) # Normalisation with windows is needed
    if corrtype=='Cross':
        S = 0 # noise drops out in cross-correlation
        F1k = np.fft.rfftn(w1*f1)
        Wk2 = np.fft.rfftn(w2*W2)
        F2k = np.fft.rfftn(w2*f2) - Ngal*Wk2
        pkspec = np.real( F1k * np.conj(F2k) )
        return Vcell / np.sum(w1*w2*W2) * (pkspec-S) * 1/Ngal # Only normalisation with galaxy window is needed

def binpk(pkspec,dims,kbins,kcuts=None,FullPk=False,doindep=True):
    '''Bin 3D power spectrum in angle-averaged bins'''
    lx,ly,lz,nx,ny,nz = dims[:6]
    kspec,muspec,indep = getkspec(dims,FullPk)
    if kcuts is not None: # Remove kspec outside kcut range to exclude from bin average:
        kperp,kpara,indep_perp,indep_para = getkspec2D(dims,FullPk)
        kperpcutmin,kperpcutmax,kparacutmin,kparacutmax = kcuts
        kspec[(kperp<kperpcutmin)] = np.nan
        kspec[(kperp>kperpcutmax)] = np.nan
        kspec[(kpara<kparacutmin)] = np.nan
        kspec[(kpara>kparacutmax)] = np.nan
    if doindep==True:
        pkspec = pkspec[indep==True]
        kspec = kspec[indep==True]
    ikbin = np.digitize(kspec,kbins)
    nkbin = len(kbins)-1
    pk,k,nmodes = np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin,dtype=int)
    for ik in range(nkbin):
        nmodes[ik] = int(np.sum(np.array([ikbin==ik+1])))
        k[ik] = np.mean( kspec[ikbin==ik+1] ) # average k-bin value for notional k
        if (nmodes[ik] > 0): #if nmodes==0 for this k then remains Pk=0
            pk[ik] = np.mean(pkspec[ikbin==ik+1])
    return pk,k,nmodes

def binpk2D(pkspec,dims,kperpbins,kparabins,FullPk=False,doindep=True):
    kperpspec,kparaspec,indep_perp,indep_para = getkspec2D(dims,FullPk=FullPk)
    if doindep==True: # Identify and remove non-independent modes
        pkspec = pkspec[(indep_perp==True) & (indep_para==True)]
        kperpspec = kperpspec[(indep_perp==True) & (indep_para==True)]
        kparaspec = kparaspec[(indep_perp==True) & (indep_para==True)]
    # Get indices where kperp and kpara values fall in bins
    ikbin_perp = np.digitize(kperpspec,kperpbins)
    ikbin_para = np.digitize(kparaspec,kparabins)
    lx,ly,lz,nx,ny,nz = dims[:6]
    nkperpbin,nkparabin = len(kperpbins)-1,len(kparabins)-1
    pk2d,k2d,nmodes2d = np.zeros((nkparabin,nkperpbin)),np.zeros((nkparabin,nkperpbin)),np.zeros((nkparabin,nkperpbin),dtype=int)
    for i in range(nkperpbin):
        for j in range(nkparabin):
            ikmask = (ikbin_perp==i+1) & (ikbin_para==j+1) # Use for identifying all kperp,kpara modes that fall in 2D k-bin
            nmodes2d[j,i] = int(np.sum(np.array([ikmask])))
            k2d[j,i] = np.mean( np.sqrt( kperpspec[ikmask]**2 + kparaspec[ikmask]**2 )  ) # average k-bin value for notional kperp, kpara combination
            if (nmodes2d[j,i] > 0):
                # Average power spectrum into (kperp,kpara) cells
                pk2d[j,i] = np.mean(pkspec[ikmask])
    return pk2d,k2d,nmodes2d

def getkspec(dims,FullPk=False,decomp=False):
    '''Obtain 3D grid of k-modes'''
    lx,ly,lz,nx,ny,nz = dims[:6]
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    if FullPk==True or decomp==True: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)
    else: kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1]
    if decomp==True: # Return coordinate tuple (kx,ky,kz) at every point on grid
        kxi,kyj,kzk = np.meshgrid(kx,ky,kz, indexing='ij')
        kspec = np.array([kxi,kyj,kzk])
        kspec = np.swapaxes(kspec,0,1)
        kspec = np.swapaxes(kspec,1,2)
        kspec = np.swapaxes(kspec,2,3)
        return kspec
    indep = getindep(nx,ny,nz)
    indep[0,0,0] = False
    if FullPk==True:
        indep = fthalftofull(nx,ny,nz,indep)
    kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
    kspec[0,0,0] = 1 # to avoid divide by zero error
    muspec = np.absolute(kz[np.newaxis,np.newaxis,:])/kspec
    muspec[0,0,0] = 1 # divide by k=0, means mu->1
    kspec[0,0,0] = 0 # reset
    return kspec,muspec,indep

def getkspec2D(dims,do2D=False,FullPk=False):
    '''Obtain two 3D arrays specifying kperp and kpara values at every point in
    pkspec array - if do2D==True - return 2D arrays of kperp,kpara'''
    lx,ly,lz,nx,ny,nz = dims[:6]
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    kperp = np.sqrt(kx[:,np.newaxis]**2 + ky[np.newaxis,:]**2)
    if FullPk==False:
        kpara = np.abs( 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1] )
        kperpspec = np.reshape( np.repeat(kperp,int(nz/2)+1) , (nx,ny,int(nz/2)+1) )
    if FullPk==True:
        kpara = np.abs( 2*np.pi*np.fft.fftfreq(nz,d=lz/nz) )
        kperpspec = np.reshape( np.repeat(kperp,nz) , (nx,ny,nz) )
    kparaspec = np.tile(kpara,(nx,ny,1))
    indep = getindep(nx,ny,nz)
    indep[0,0,0] = False
    if FullPk==True:
        indep = fthalftofull(nx,ny,nz,indep)
    indep_perp,indep_para = np.copy(indep),np.copy(indep)
    indep_perp[kperpspec==0] = False
    indep_para[kparaspec==0] = False
    kparaspec[0,0,0],kperpspec[0,0,0] = 0,0
    return kperpspec,kparaspec,indep_perp,indep_para

def getindep(nx,ny,nz):
    '''Obtain array of independent 3D modes'''
    indep = np.full((nx,ny,int(nz/2)+1),False,dtype=bool)
    indep[:,:,1:int(nz/2)] = True
    indep[1:int(nx/2),:,0] = True
    indep[1:int(nx/2),:,int(nz/2)] = True
    indep[0,1:int(ny/2),0] = True
    indep[0,1:int(ny/2),int(nz/2)] = True
    indep[int(nx/2),1:int(ny/2),0] = True
    indep[int(nx/2),1:int(ny/2),int(nz/2)] = True
    indep[int(nx/2),0,0] = True
    indep[0,int(ny/2),0] = True
    indep[int(nx/2),int(ny/2),0] = True
    indep[0,0,int(nz/2)] = True
    indep[int(nx/2),0,int(nz/2)] = True
    indep[0,int(ny/2),int(nz/2)] = True
    indep[int(nx/2),int(ny/2),int(nz/2)] = True
    return indep

def fthalftofull(nx,ny,nz,halfspec):
    '''Fill full transform given half transform'''
    fullspec = np.empty((nx,ny,nz))
    ixneg,iyneg,izneg = nx-np.arange(nx),ny-np.arange(ny),nz-np.arange(int(nz/2)+1,nz)
    ixneg[0],iyneg[0] = 0,0
    fullspec[:,:,:int(nz/2)+1] = halfspec
    fullspec[:,:,int(nz/2)+1:nz] = fullspec[:,:,izneg][:,iyneg][ixneg]
    return fullspec

def fthalftofull2(nx,ny,nz,halfspec1,halfspec2):
    fullspec = np.empty((nx,ny,nz))
    ixneg,iyneg,izneg = nx-np.arange(nx),ny-np.arange(ny),nz-np.arange(int(nz/2)+1,nz)
    ixneg[0],iyneg[0] = 0,0
    fullspec[:,:,:int(nz/2)+1] = np.real(halfspec1*np.conj(halfspec2))
    fullspec[:,:,int(nz/2)+1:nz] = fullspec[:,:,izneg][:,iyneg][ixneg]
    return fullspec

def getpkconv(pkspecmod,dims,w1=None,w2=None,W1=None,W2=None):
    '''Convolve model power spectrum with weights and window functions'''
    # w1/w2: optional field weights
    # W1/W2: optional survey selection functions
    lx,ly,lz,nx,ny,nz = dims[:6]
    pkspecmod[0,0,0] = 0
    # Apply default unity weights/windows if none given:
    if w1 is None: w1 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if w2 is None: w2 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if W1 is None: W1 = np.ones((nx,ny,nz))# set to unity if no weights provided
    if W2 is None: W2 = np.ones((nx,ny,nz))# set to unity if no weights provided
    W1,W2 = W1/np.sum(W1),W2/np.sum(W2) # Normalise window functions so sum(W)=1
    Wk1 = np.fft.rfftn(w1*W1)
    Wk2 = np.fft.rfftn(w2*W2)
    Wk = fthalftofull2(nx,ny,nz,Wk1,Wk2) ; del Wk1; del Wk2
    # FFT model P(k) and W(k) (despite already being in Fourier space) in order to
    #   use convolution theorem and multiply Fourier transforms together:
    pkspecmodFT = np.fft.rfftn(pkspecmod)
    Wk1FT = np.fft.rfftn(Wk); del Wk
    pkcongrid = np.fft.irfftn(pkspecmodFT*Wk1FT) # Inverse Fourier transform
    return pkcongrid / ( nx*ny*nz * np.sum(w1*w2*W1*W2) )
