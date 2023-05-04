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

where $\bf{v}$ denotes the event vertex position, $\bf{h_i}$ the position of the $i$-th hit PMT, and $c^{'}$ the group velocity of Cherenkov light in water. Ideally, $t_{res,i}$ has the common value to all hit PMTs. The truth emission time here is 0. We can then construct a PDF of the hit-timing residual using many events by fitting the distribution. This will be the PDF in which we sample from to attempt to maximize our "observed" hit times. Upon fitting many different functions to the hit times, we elect to use a non-central student's t continuous PDF. It fit the low-energy data well (from 5-30 MeV), and it has a simple form that is easy to work with:

If $Y$ is a standard normal r.v. and $V$ is an independent chi-square random variable with $k$ degrees of freedom, then:

$$
X = \frac{Y + c}{\sqrt{V/k}}
$$

where $c$ is the noncentrality parameter. In scipy, the parameters returned for scipy.stats.nct are (df, nc, loc, scale).

Having fit the PDF above, we can use our favorite maximium likelihood function to find the associated likelihood of a given test vertex. Currently, the MCMC algorithm uses Super-K's BONSAI likelihood function:

$$
\ln{L}(x,t_0) = \ln{\prod_{i=1}^{N} P(\Delta t_i (x))} = \sum_{i=1}^{N} \ln{P(t_i - tof_i (x) - t_0)}
$$

where the hit time residual $\Delta t_i (x)$ is again given by:

$$
\Delta t_i (x) = t_i - tof_i (x) - t_0
$$

where $x$ is the test vertex, $t_i$ is the hit time at the $i$-th PMT, $t_0$ is the emission time (not 0 in this case, since it is an unknown) and $tof_i = |x_i - x|/c_{water}$ is the time of flight from the reconstructed vertex to the PMT vertex for hit i. In the future, a more appropriate likelihood function can be used for ANNIE.

The overall strategy for finding the most likely vertex, given the data, is then to sample guess verticies. This is done by calculating $\Delta t_i (x)$ for each PMT hit time to get a collection of time residuals for a given guess vertex. Then, sample the PDF for each hit time residual to get the associated probability. Use all the hit time probabilities to calculate the log(likelihood) for a test vertex. Ideally, the reconstructed vertex that is closest to the truth vertex will yield the highest likelihood.

A host of different numerical strategies can be employed to search for the most likely vertex. The numerical method used for this reconstruction algorithm relies on an affine-invariant markov chain monte carlo method. Emcee (https://emcee.readthedocs.io/en/stable/) is a pure-python implementation of the Affine-Invariant Ensemble Sampler (AIES) proposed by Goodman and Weare 2010. AIES utilizies an ensemble of walkers. The beauty of AIES is that its essentially a smart, randomized advanced grid search where the walkers talk to one another to sample the probability space. The walkers move via "stretch moves", where the walker's motions are based on the current positions of the other walkers. 

To update a given walker $X_k$, the method selects (at random) another walker $X_j$. A proposal point is generated along the straight line (in the parameter space) connecting the two walkers. Moving the main walker to the proposed, new position depends upon some acceptance probability, which is based on the ratio of the target probability densities at the current and proposal points. If the proposal point is more likely, the main walker moves. This is done in sequence, with many walkers simulataneously; thus, if one walker samples a region in probability space that has a high likelihood, it will pull the other walkers towards it. The path followed by the ensemble is markov, so the results stay unbiased. This numerical approach also avoids getting trapped in local minimas by effecivtely sampling more of the parameter space. 

## Hit Filtering

to be updated

## How to use (for MC)

(Pre-steps, in ToolAnalysis and using .C scripts to extract the relevant hits information)
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

(Running the code)
1. Modify `emcee_lowE_reco.py` for the correct path names and other configuration information. This parametrization is at the top of the script.
2. Run the code `python3 emcee_lowE_reco.py`. It will perform hit filtering and reconstruct the verticies. You will eventually be left with the corresponding .dat files and (if specified) emcee plots depicting the walker distributions, in comparison with the truth vertex info.

##
Files and Directories included:
- `FullTankPMTGeometry.csv` contains the geometry, locations, type, and overall information of the PMTs and is important for loading in the detector geometry and establishing the priors (initial constraints on where the vertices could be).
* `\emcee_files` is where the final reconstructed vertex positions and errors will be placed. It contains some example .dat files produced from the main reconstruction algorithm.
* `\emcee_plots` is the directory where (if specified) the emcee corner plots will be produced. These show the walker distributions in each dimension, along with the associated truth (MC) vertex information.
* `PDF.dat` is the output of the 'fit_PDF_residual.py' script which generates the corresponding parameters for the hit-timing residual PDF (approximately non-central student's t).
* `\WCSim Data` contains the MC hit root trees from ToolAnalysis, and is where the .dat generated files (from the root trees) will be produced, and called from in the main algorithm.
* `reco_dependencies.sh` is a bash script containing commands for downloading the required python dependencies. Comment out the dependencies you already have. Have not tested this.
