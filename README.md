# gridimp
**grid**ded **i**ntensity **m**apping **p**ower

## Introduction:
Toolkit for regridding line intensity maps onto a Cartesian grid on which fast fourier transforms can be run for analysing n-point clustering statistics (such as the power spectrum) in Fourier space. The pipeline provides two novel features: 
 - a validated Monte-Carlo style resampling from map (R.A., Dec. Frequency) space onta a Cartesian (x, y, z in [Mpc/h]) grid,
 - option of higher-order interpolation of the sampling particles (using ``pmesh``), which mitigates aliasing effects. 

The algorithms in this package are discussed in detail in the accompanying paper (https://arxiv.org/pdf/23??.????.pdf), where validation on a suite of simulations is presented.

## Installation:
1. Clone this repo `git clone https://github.com/stevecunnington/gridimp`.
2. Change directory to the cloned gridimp repo.
3. Activate a python environment with the required packages (see **full list of dependencies** in the `environments.yaml` file). Simplest way to create and activate a new anaconda environment with the necessary packages is with:
```
conda env create -f environment.yml
conda activate gridimp
```
4. Install the repo with:
```
python setup.py install
```

## Getting started:
```
Some lines of code demonstrating how to regrid a LIM
```
See ``scripts`` folder for more in-depth code which reproduce the results in the accompanying paper (https://arxiv.org/pdf/23??.????.pdf).


## Acknowledgement:
If you use any of the code in gridimp, we kindly ask that you cite the accompanying paper (https://arxiv.org/pdf/23??.????.pdf).

## Contact:
Feel free to get in contact with any comments, suggestions, or bug reports at:
<br /> **Steve Cunnington** [steven.cunnington@manchester.ac.uk],
<br /> alternatively, please open a New issue from the **Issues** tab.
