# MCMC_reco

A low energy reconstruction algorithm for WCSim-generated MC events. Currently, no low energy reconstruction algorithm has been specifically developed for ANNIE; a working, reliable low energy reconstruction algorithm is needed for localizing neutral current events within the fiducial volume. It also can be applied to reconstructing the positions of deexcitation gammas from neutron-Gd capture.

Will eventually be integrated into the ToolAnalysis framework.

The main algorithm takes information extracted from .ROOT files produced using Tools in ToolAnalysis and performs an affine invariant Markov-Chain Monte Carlo (MCMC) ensembler sampler to reconstruct the most likely vertex position in (x,y,z,ct).

Hit filtering is done prior to ensure the hits fed into MCMC are reflective of a single interaction vertex. Only hit timing is utilized. Eventually a charge-based likelihood can be folded in, which is more relevant for higher energy events.

## Likelihood maximization with MCMC

For a given neutrino event in the ANNIE tank, there will be a corresponding set of PMT hits. The hit times of the PMTs fold in the effects of the interaction topology, but also include PMT timing features, photocoverage, and scattering + reflection in the detector medium. As a result, a hit timing residual PDF can be constructed that details your detector's response to a given neutrino event. For this reconstruction algorithm, MC "truth" vertex information is used to calculate the hit timing residual for each hit in a given event:

$$
t_{res,i}(v) \equiv t_i - \frac{|\bf{v} - \bf{h_i}|}{c^{'}} - t_0
$$

where $\bf{v}$ denotes the event vertex position, $\bf{h_i}$ the position of the $i$-th hit PMT, $c^{'}$ the group velocity of Cherenkov light in water, and $t_0$ is the emission/interaction time. Ideally, $t_{res,i}$ has the common value to all hit PMTs. We can then construct a PDF of the hit-timing residual using many events by fitting the distribution using ```fit_PDF_residual.py```. This will be the PDF in which we sample from to attempt to maximize over our "observed" hit times.

Having fit the PDF above with some function, we can use our favorite maximium likelihood function to find the associated likelihood of a given test vertex. Currently, the MCMC algorithm uses Super-K's BONSAI likelihood function:

$$
\ln{L}(x,t_0) = \ln{\prod_{i=1}^{N} P(\Delta t_i (x))} = \sum_{i=1}^{N} \ln{P(t_i - tof_i (x) - t_0)}
$$

where the hit time residual $\Delta t_i (x)$ is again given by:

$$
\Delta t_i (x) = t_i - tof_i (x) - t_0
$$

where $x$ is the test vertex, $t_i$ is the hit time at the $i$-th PMT, $t_0$ is the emission time and $tof_i = |x_i - x|/c_{water}$ is the time of flight from the reconstructed vertex to the PMT vertex for hit i. In the future, a more appropriate likelihood function can be used for ANNIE.

The overall strategy for finding the most likely vertex, given the data, is then to sample guess verticies. This is done by calculating $\Delta t_i (x)$ for each PMT hit time to get a collection of time residuals for a given guess vertex. Then, sample the PDF for each hit time residual to get the associated probability. Use all the hit time probabilities to calculate the log(likelihood) for a test vertex. Ideally, the reconstructed vertex that is closest to the truth vertex will yield the highest likelihood.

A host of different numerical strategies can be employed to search for the most likely vertex. The numerical method used for this reconstruction algorithm relies on an affine-invariant markov chain monte carlo method. Emcee (https://emcee.readthedocs.io/en/stable/) is a pure-python implementation of the Affine-Invariant Ensemble Sampler (AIES) proposed by Goodman and Weare 2010. AIES utilizies an ensemble of walkers. The beauty of AIES is that it's essentially a smart, randomized advanced grid search where the walkers talk to one another to sample the probability space. The walkers move via "stretch moves", where the walker's motions are based on the current positions of the other walkers. 

To update a given walker $X_k$, the method selects (at random) another walker $X_j$. A proposal point is generated along the straight line (in the parameter space) connecting the two walkers. Moving the main walker to the proposed, new position depends upon some acceptance probability, which is based on the ratio of the target probability densities at the current and proposal points. If the proposal point is more likely, the main walker moves. This is done in sequence, with many walkers simulataneously; thus, if one walker samples a region in probability space that has a high likelihood, it will pull the other walkers towards it. The path followed by the ensemble is markov, so the results stay unbiased. This numerical approach also avoids getting trapped in local minimas by using multiple samplers, all of which sample more of the parameter space at random. In addition, a prior is used to exclude some of the parameter space (for this, a simple "don't explore more than x meters away from the detector tank" is employed).

For a given event in the tank, this method then returns distributions of all the walkers in the 4D parameter space after all steps are complete. The most common walker position at the end of the process is taken to be the reconstructed vertex.

## Hit Filtering

To ensure the vertex reconstruction is not using light from reflections, scattering, etc... hit cleaning and filtering based on hit times is used. No current cuts are employed that consider the charge of the hits, although it may be useful to make a cut on hits below 1pe for example to limit reflections. As a particle interacts in the water, it will produce light from multiple points along the track/shower. As this track/shower is on the order of a fraction of a meter, there is a bias that is likely introduced in the reconstruction given how small the ANNIE tank is. For our purposes, we will try our best to take the very first light from the initial interaction, peel away light due to scattering and/or reflections, and perform the reconstruction on the hits that pass our cuts.

To try and remove reflections, the hit cleaning considers the first 4 hit times in a cluster. Later hits are rejected/accepted if they are within 10ns of the average of the 4 initial hit times. In the future, a charge-based rejection criteria can be used to try and further/more efficiently eliminate reflections. Hits that pass this 10ns criteria are carried to the next stage of hit filtering.

All hits within this 10ns threshold are then checked if they are causally disconnected from the first 4 hits. Ideally, we want each hit to be causally independent from all other hits, i.e. they come from roughly the same interaction point. If light scatters along its path from the vertex to PMT, it will take additional time on top of the time of flight. Other light may be produced later on as the particle propagates/interacts with the medium. Hits to be considered beyond the first 4 hits in a cluster must be seperated in time less than the light travel distance from that hit PMT to the other PMTs. If that is the case, we can say the light is likely from the same interaction point. Causal independence is checked for additional PMT hits with each of the initial 4. It is assumed the initial 4 hits are from the same interaction point (assuming they pass the 10ns cut above).

In BONSAI, causual independence is checked between each possible PMT pair; this technique was a little too computationally expensive for clusters with many events (say over 10), and since the emcee reconstruction takes long too, I adopted the technique above.

In the future, it should be checked if the code can correct for any systematic shift along the track/shower direction, as the track/shower cannot be approximated as a point source given how small ANNIE is.

The final, filtered set of hits are passed to the reconstruction. The reconstruction will skip clusters/events with less than 4 final filtered hits, as 4 hits in principle is needed to solve for the vertex position (since we are using MCMC, technically it can be solved via sampling with less than 4 hits, but the errors are large).

## How to use (for MC)

(Pre-steps, including ToolAnalysis usage)
1. Run WCSim on some simulated particle interaction (or use an exisiting WCSim output root file).
2. Run the BeamClusterMC ToolChain withinin ToolAnalysis to extract information from the WCSim output .ROOT file. The clusterization tools will filter hits within an alloted time interval (defined as a cluster) and pass these hits along as an event to the PhaseIITreeMaker tool. This tool will create a .ROOT file containing trees (cluster-level and raw hits) and histograms of various parameters of the events.
3. Store the .ROOT output file from ToolAnalysis into the `WCSim_Data/` folder.
4. Modify `PDF.dat` to include the fit parameters of the hit residual data. Running `fit_PDF_residual.py` will fit the hit timing residual data with a PDF, and write to that .dat file. 

(Running the code)
1. Modify `emcee_lowE_reco.py` for the correct path names and other configuration information. The parametrization is at the top of the script.
2. Run the code `python3 emcee_lowE_reco.py <starting_event> <ending_event>`. The two inputs provided are the starting and ending event number. The code will perform hit filtering and reconstruct the verticies. You will eventually be left with the corresponding .dat files and (if specified) emcee plots depicting the walker distributions, in comparison with the truth vertex info.

##
Files and Directories included:
* `output/` is where the ntuple containing the final reconstructed vertex positions and errors will be placed. It is produced from the main reconstruction algorithm.
* `emcee_plots/` is the directory where (if specified) the emcee corner plots will be produced. These show the walker distributions in each dimension, along with the associated truth (MC) vertex information.
* `PDF.dat` is the output of the 'fit_PDF_residual.py' script which generates the corresponding parameters for the hit-timing residual PDF.
* `WCSim_Data/` contains the MC hit root trees from ToolAnalysis.
* `reco_dependencies.sh` is a bash script containing commands for downloading the required python dependencies. Comment out the dependencies you already have. Have not tested this.
