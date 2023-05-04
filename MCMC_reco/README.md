# MCMC_reco

A low energy reconstruction algorithm for WCSim-generated MC events. Currently, no low energy reconstruction algorithm has been specifically developed for ANNIE; a working, reliable low energy reconstruction algorithm is needed for localizing neutral current events within the fiducial volume. It also can be applied to reconstructing the positions of deexcitation gammas from neutron-Gd capture.

Will eventually be integrated into the ToolAnalysis framework.

The main algorithm takes information extracted from .ROOT files produced using Tools in ToolAnalysis and performs an affine invariant Markov-Chain Monte Carlo (MCMC) ensembler sampler to reconstruct the most likely vertex position in (x,y,z,ct).

Hit filtering is done prior to ensure the hits fed into MCMC are reflective of a single interaction vertex. Only hit timing is utilized. Eventually a charge-based likelihood can be folded in, which is more relevant for higher energy events (> 30 MeV).

## Likelihood maximization with MCMC

For a given neutrino event in the ANNIE tank, there will be a corresponding set of PMT hits. The hit times of the PMTs fold in the effects of the interaction topology, but also include PMT timing features, photocoverage, and scattering + reflection in the detector medium. As a result, a hit timing residual PDF can be constructed that details your detector's response to a given neutrino event. For this reconstruction algorithm, MC "truth" vertex information is used to calculate the hit timing residual for each hit in a given event:

$$
t_{res,i}(v) \equiv t_i - \frac{|\bf{v} - \bf{h_i}|}{c^{'}}
$$

where $\bf{v}$ denotes the event vertex position, $\bf{h_i}$ the position of the $i$-th hit PMT, and $c^{'}$ the group velocity of Cherenkov light in water.

Ideally, $t_{res,i}$ has the common value to all hit PMTs. The truth emission time here is 0.

We can then construct a PDF of the hit-timing residual using many events. The will be the PDF in which we sample from to attempt to maximize our "observed" hit times.


The reconstruction algorithm works to maximize the likelihood of a hypothesis vertex. The likelihood is calculated 



hit-time residual for a given event - this involves finding the reconstructed vertex (x,y,z,ct) with the highest likelihood, according to some "truth" timing residual information. For this, we rely on the timing residuals WRT the MC truth vertex



## How to use (for MC)

1. Run the BeamClusterMC ToolChain withinin ToolAnalysis to extract information from the WCSim output .ROOT file. The clusterization tools will filter hits within an alloted time interval (defined as a cluster) and pass these hits along as an event to the PhaseIITreeMaker tool. This tool will create a .ROOT file containing trees (cluster-level and raw hits) and histograms of various parameters of the events.
2. Store the .ROOT output file from ToolAnalysis into the `/MC_Data` folder.
3. Using the .C scripts located in the main directory (`charge_parameters.C`, `cluster_hits.C`, `true_MC.C`), extract hits and cluster information from each event into .dat files to be read by the jupyter notebook event display.
      - For each .C file (3 in total), adjust the filemame passed to the TFiles and the name/path of the output `.dat` file. `charge_parameters.C` reads the event-level information (charge balance, maximum p.e., number of hits, etc..) from the TankClusterTree. `cluster_hits.C` reads the hit-level information (multiple hits per event) (hit time, hit location, p.e.'s, etc..) from the TankClusterTree. `true_MC.C` reads the truth event-level information from the raw TriggerTree. 
      - Run the scripts (3 in total):
```
root -l scriptname.C
```
You will have produced corresponding `.dat` files for each script in the `/Extracted_Data` folder.

This will be folded into a single bash script in the future.

4. 

##
Files and Directories included:
- `FullTankPMTGeometry.csv` contains the geometry, locations, type, and overall information of the PMTs and is important for loading in the detector geometry and establishing the priors (initial constraints on where the vertices could be).
* `\emcee_files` is where the final reconstructed vertex positions and errors will be placed. It contains some example .dat files produced from the main reconstruction algorithm.
* `\emcee_plots` is the directory where (if specified) the emcee corner plots will be produced. These show the walker distributions in each dimension, along with the associated truth (MC) vertex information.
* `PDF.dat` is the output of the 'fit_PDF_residual.py' script which generates the corresponding parameters for the hit-timing residual PDF (approximately non-central student's t).
* `\WCSim Data` contains the MC hit root trees from ToolAnalysis, and is where the .dat generated files (from the root trees) will be produced, and called from in the main algorithm.
