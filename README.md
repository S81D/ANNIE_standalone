# Standalone ANNIE Analysis

This repository contains various subdirectories consisting of standalone analysis tools and scripts created and implemented outside the primary analysis framework for ANNIE, ToolAnalysis. Eventually, some of these analyses will be converted into Tools within ToolAnalysis.

## Subdirectories

1. **pyANNIEEventDisplay**
     - Python Event Display for visualizing events within the ANNIE detector. Currently implemented for MC low-energy events (no MRD/FMV).

2. **MCMC_reco**
     - Python-based reconstruction algorithm for low energy MC events generated in WCSim. The reconstruction algorithm relies on the affine-invariant Markov-Chain Monte Carlo (MCMC) ensembler sampler to reconstruct the most likely vertex position in (x,y,z,ct). MCMC carried out through the python module, emcee.
