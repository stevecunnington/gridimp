# gridimp
**grid**ded **i**ntensity **m**apping **p**ower

## Introduction:
Toolkit for regridding line intensity maps onto a Cartesian grid on which fast Fourier transforms can be run for analysing n-point clustering statistics (such as the power spectrum) in Fourier space. The pipeline provides two novel features:
 - a validated Monte-Carlo style resampling from map (R.A., Dec. Frequency) space onto a Cartesian (x, y, z in [Mpc/h]) grid,
 - option of higher-order interpolation of the sampling particles (using ``pmesh``), which mitigates aliasing effects.

The algorithms in this package are discussed in detail in the accompanying paper (https://arxiv.org/pdf/2312.07289.pdf), where validation on a suite of simulations is presented.

## Installation:
1. Clone this repo `git clone https://github.com/stevecunnington/gridimp`.
2. Change directory to the cloned gridimp repo. ```cd gridimp```
3. Activate a python environment with the required packages (see **full list of dependencies** in the `environments.yml` file). The simplest way to create and activate a new anaconda environment with the necessary packages is with:
```
conda env create -f environment.yml
conda activate gridimp
```
4. Install the repo with:
```
python setup.py install
```
5. Validate the package installation by running the test demo script:
```
python examples/toy.py
```

* For installation on ilifu [https://www.ilifu.ac.za/]
Since this package is frequently used on ilifu, and users have experienced problems with stalling following the above installation steps, we share the following steps using miniforge [https://github.com/conda-forge/miniforge] which have helped overcome stalling issues in the environment build.

a. First install conda miniforge with
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```
b. Then use ``mamba`` to create the environment in step 3.:
```
mamba env create -f environment.yml
conda activate gridimp
```
c. Follow steps 4. and 5. as before.

## Getting started:

* #### Cartesian lognormal mock:
Generate a fast cubic lognormal mock over n=128^3 cells with size l=500^3(Mpc/h)^3:
```
from gridimp import cosmo
from gridimp import mock

cosmo.SetCosmology(z=1) # set cosmology
Pmod = cosmo.GetModelPk(z=1) # obtain matter power spectrum

l = 500 # length of box down one side [Mpc/h]
n = 128 # number of cells down one side
dims = [l,l,l,n,n,n,0,0,0] # lengths,cells,origins for each dimension

delta_0 = mock.Generate(Pmod,dims) # generate LIM mock (default b=1,T=1)
```

* #### HEALPix sky maps for simulated survey:
By defining input simulated survey parameters, a lognormal mock can be generated onto a Cartesian grid calculated to fully enclose the survey footprint.

```
### Define survey parameters:
ramin,ramax = 10,30 # [deg]
decmin,decmax = 10,30  # [deg]
numin,numax = 900,1110 # [MHz]
nnu = 60 # number of frequency channels
nside = 128 # HEALPix resolution: determines pixel size
n0 = 256 # n0^3 will be number of cells for input grid cube

### Initialise healpix and get sky coordinates for voxels covering survey:
from gridimp import grid
ra,dec,nu,dims_0 = grid.init_healpix(nside,ramin,ramax,decmin,decmax,numin,numax,nnu,n0)

delta_0 = mock.Generate(Pmod,dims_0) # generate LIM mock (default b=1,T=1)

# Create healpy sky maps for each frequency channel:
m = grid.lightcone_healpy(delta_0,dims_0,ra,dec,nu,nside)
```

* #### Resample sky maps to Cartesian grid on which FFT can be run:
```
n = 64 # n^3 will be number of cells for out FFT grid
Np = 5 # number of sampling particles per map voxel for regridding
window = 'ngp' # mass assignment function
compensate = True # correct for interpolaton effects at field level
interlace = False # interlace FFT fields

# Regrid to FFT:
from astropy import units as u
from gridimp import line
ra_p,dec_p,nu_p,pixvals = grid.SkyPixelParticles_healpy(ra,dec,nu,nside,m,Np=Np)
xp,yp,zp = grid.SkyCoordtoCartesian(ra_p.to(u.deg).value,dec_p.to(u.deg).value,line.nu21cm_to_z(nu_p),ramean_arr=ra.to(u.deg).value,decmean_arr=dec.to(u.deg).value,doTile=False)
dims_fft = grid.comoving_dims(ra,dec,nu,nside,(n,n,n))
delta_G,W,counts = grid.mesh(xp,yp,zp,pixvals,dims_fft,window,compensate,interlace)
```

``examples/toy.py`` has a walkthrough of simulation generation -> Healpix sky mapping -> Cartesian regridding -> power spectrum estimation.

``scripts`` folder has the original in-depth code used to generate the main results in the accompanying paper (https://arxiv.org/pdf/2312.07289.pdf).


## Acknowledgement:
If you use any of the code in gridimp, we kindly ask that you cite the accompanying paper (https://arxiv.org/pdf/2312.07289.pdf).

## Contact:
Feel free to get in contact with any comments, suggestions, or bug reports at:
<br /> **Steve Cunnington** [steven.cunnington@manchester.ac.uk],
<br /> alternatively, please open a New issue from the **Issues** tab.
