# gridimp
**grid**ded **i**ntensity **m**apping **p**ower

## Introduction:
Toolkit for regridding line intensity maps onto a Cartesian grid on which fast fourier transforms can be run for analysing n-point clustering statistics (such as the power spectrum) in Fourier space. The pipeline provides two novel features: 
 - a validated Monte-Carlo style resampling from map (R.A., Dec. Frequency) space onta a Cartesian (x, y, z in [Mpc/h]) grid,
 - option of higher-order interpolation of the sampling particles (using ``pmesh``), which mitigates aliasing effects. 

The algorithms in this package are discussed in detail in the accompanying paper (https://arxiv.org/pdf/2212.08056.pdf), where validation on a suite of simulations is presented.

## Installation:
1. Activate a python environment with the required packages (specified in the `environments.yaml` file.) Simplest way to create and activate a new anaconda environment is with:
```
conda env create -f environment.yml
conda activate gridimp
```
