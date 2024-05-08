#!/bin/bash

# some python dependencies that are needed for the main scripts
# comment out the ones that already exist on your computer
# (assumes that you have pip installed)

# Read and write root files
pip install uproot

# PDF fitting
#python3 -m pip install statsmodels
# or
pip install statsmodels
pip install pandas
pip install scipy

# emcee reco
pip install corner
pip install emcee
pip install tqdm
