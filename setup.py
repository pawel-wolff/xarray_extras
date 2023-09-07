#!/usr/bin/env python

from setuptools import setup

setup(
    name='xarray-extras',
    version='0.1',
    description='xarray extensions',
    author='Pawel Wolff',
    author_email='pawel.wolff@aero.obs-mip.fr',
    packages=[
        'xarray_extras',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'xarray>=0.19',
        'dask',
    ],
)
