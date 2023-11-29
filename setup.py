import json
import os
import sys
from setuptools import setup

setup_args = {
    'name': 'gridimp',
    'author': 'Steve Cunnington',
    'url': 'https://github.com/stevecunnington/gridimp',
    'license': 'MIT',
    'version': '0.0.9',
    'description': 'Toolkit for regridding line intensity maps onto a Cartesian grid on which fast fourier transforms can be run for analysing n-point clustering statistics (such as the power spectrum) in Fourier space.',
    'packages': ['gridimp'],
    'package_dir': {'gridimp': 'gridimp'},
    'install_requires': [
        'python=3.7',
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
        'astropy-healpix',
        'pyfftw',
        'classylss',
        'pmesh'
    ],
    'extras_require': {'fgextras': ['healpy']},
    'include_package_data': True,
    'zip_safe': False
}

if __name__ == '__main__':
    setup(**setup_args)
