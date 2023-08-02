################################################################
# MCMC-based Python Reconstruction for low energy ANNIE Events #
# ------------------------------------------------------------ #
#   Using the hit time residuals in the tank (PMT only) and a  # 
#    BONSAI likelihood function, perform an affine-invariant   #
#      Markov-Chain Monte Carlo (MCMC) ensembler sampler to    #
#   reconstruct the most likely vertex position in (x,y,z,ct). #
#       MCMC carried out through the python module, emcee.     #
# ------------------------------------------------------------ #
#          Author: Steven Doran   Date: April 2023             #
################################################################

# import packages

import sys                        # allow script to take input arguments
import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import nct       # PDF for the hit-timing residuals
import emcee                      # python implementation of the Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler
import corner                     # necessary for generating diagnostic plots for emcee
import pandas as pd
import uproot3 as uproot          # read root files directly in python


''' Things to check before you run this code:
    1. Is there a root file in /WCSim_data?
    2. Did you run fit_PDF_residual.py to generate a PDF fit to the time-residual data?
'''



''' ###################################################### '''
''' # ------ Parameterization - Requires User Input ---- # '''
''' ###################################################### '''

# 1. ---------------
file = uproot.open('WCSim_Data/10MeV/electron_swarm_10MeV.ntuple.root')


# 2. ---------------
# Number of events/clusters we are performing the reconstruction on
# (variables passed by user input)
start_event = int(sys.argv[1])
final_event = int(sys.argv[2])


# 3. ---------------
# emcee ensemble parameters
nwalkers = 1000         # number of walkers
burn_in_steps = 25      # number of burn-in steps to "feel out" the probability space
n_steps = 50            # number of steps for walkers in the primary run
prg_bar = True         # enable/disable progress bar. For grid submission, set = False

save_picture = False    # specify whether you want to save the .png corner plots
save_first_ten = False  # often useful for diagnostics, see if the reco algorithm is working before doing more events
save_frequency = 10     # can be used as code update, if save_picture is enabled, the code will save every (save_frequency)-th picture
                        # set to 1 if you want all pictures saved 


''' ###################################################### '''


# -------------------------------------
# Part 1 - Load in Data

print('Loading data...')

# cluster-level information
T = file['phaseIITankClusterTree']

clustereventNumber = T['eventNumber'].array()
clusterNumber = T['clusterNumber'].array()
clusterPE = T['clusterPE'].array()
clusterMaxPE = T['clusterMaxPE'].array()
clusterChargeBalance = T['clusterChargeBalance'].array()
clusterHits = T['clusterHits'].array()
clusterTime = T['clusterTime'].array()

# Total number of clusters (could be none, one, or multiple clusters per event)
N_clusters = len(clustereventNumber)

# hits-level information
Channel = T['hitChankeyMC'].array()
hitT = T['hitT'].array()
hitX = T['hitX'].array()
hitY = T['hitY'].array()
hitZ = T['hitZ'].array()
hitPE = T['hitPE'].array()


# MC Truth information (need to sort and assign them)
MC = file['phaseIITriggerTree']

eventNumber = MC['eventNumber'].array()  # Event Number
vtX = MC['trueVtxX'].array()             # {vertex information   
vtY = MC['trueVtxY'].array()             # ..
vtZ = MC['trueVtxZ'].array()             # }
vtT = MC['trueVtxTime'].array()          # "vertex time" i.e. initial time
dirX = MC['trueDirX'].array()            # {direction vectors of primary particle
dirY = MC['trueDirY'].array()            # ..
dirZ = MC['trueDirZ'].array()            # }
Energy = MC['trueMuonEnergy'].array()    # initial energy of the primary particle
Track_Length = MC['trueTrackLengthInWater'].array()  # track length of the primary particle in water 
                                                     # (distance from start point to stop point or the
                                                     # distance from the start vertex to a tank wall (if the particle exited))
    
N_events = len(eventNumber)    # Number of WCSim generated events

# sort events that have an associated cluster event number
vtX = [vtX[int(en)]/100 for en in clustereventNumber]
vtY = [vtY[int(en)]/100 for en in clustereventNumber]
vtZ = [vtZ[int(en)]/100 for en in clustereventNumber]
# divide by 100     ^  to convert vertex units [cm] into [m]
vtT = [vtT[int(en)] for en in clustereventNumber]
dirX = [dirX[int(en)] for en in clustereventNumber]
dirY = [dirY[int(en)] for en in clustereventNumber]
dirZ = [dirZ[int(en)] for en in clustereventNumber]
Energy = [Energy[int(en)] for en in clustereventNumber]
Track_Length = [Track_Length[int(en)] for en in clustereventNumber]

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# MC Truth Vertex and direction information
origin = np.zeros([N_clusters,3])
dir_vector = np.zeros([N_clusters,3])
for i in range(N_clusters):
    origin[i][0] = vtZ[i]; dir_vector[i][0] = dirZ[i]
    origin[i][1] = vtX[i]; dir_vector[i][1] = dirX[i]
    origin[i][2] = vtY[i]; dir_vector[i][2] = dirY[i]
        
# # # # # # # # # # # # # # # # # # # # # # # # # # #
### Load in and Construct the Detector Geometry ###

# Read Geometry.csv file to get PMT location info
df = pandas.read_csv('FullTankPMTGeometry.csv')

channel_number = []; location = []; panel_number = []
x_position = []; y_position = []; z_position = []

# The LoadGeometry File does not share the same origin point as the WCSim output after ToolAnalysis.

# vertical center (y) is at a height of y = -14.46 cm
# x-axis is fine
# z-axis (beamline) is offset by 1.681 m
# tank center is therefore at (0,-14.46, 168.1) [cm]

for i in range(len(df['channel_num'])):   # loop over PMTs
    channel_number.append(df['channel_num'][i])
    location.append(df['detector_tank_location'][i])
    x_position.append(df['x_pos'][i]+0)
    y_position.append(df['y_pos'][i]+0.1446)
    z_position.append(df['z_pos'][i]-1.681)
    panel_number.append(df['panel_number'][i])

# Find the dimensions of the detector (radius, height, etc..)
height_detector = max(y_position) - min(y_position)
radius_detector = (max(z_position) - min(z_position))/2   

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

print('\nData Loaded\n')
print('#################')
print('\nNumber of Events = ', N_events)
print('Number of Clusters = ', N_clusters)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #


print('\nRunning ', (final_event - start_event), ' event/cluster(s) (start = event ', start_event, ', final = event ', final_event, ')') 


# -------------------------------------
# Part 2 - Filter hits

print('\nFiltering hits...\n')

# Hit Filterization (This can be built as a seperate tool)

c = 299792458  # [m/s]i
c = c/(4/3)    # refractive index of water

# rejection criteria
def arrival_t_criteria(ti, tj, dX):
    delta_tij = np.abs(ti - tj)
    tof = round(((np.abs(dX)/c)*1e9),0)    # PMT times are digitized in units of 2ns, so round the tof to nearest int
    if delta_tij <= tof:                   # since rounding, keep if even
        condition = True
    else:
        condition = False
    return condition


HIT = [[] for i in range(N_clusters)]        # final, filtered, reduced hits list
for i in range(len(HIT)):
    for j in range(4):
        HIT[i].append([])


a_start = start_event
a_end = final_event

for event in range(a_start,a_end):          # (how many events are you running?)
    
    print('########## Cluster ', event, ' ##########')
    
    # # # # # # # # # # # # # # # # # # #
    # remove isolated hits
    filtered_hits = [[], [], [], [], []]   # x,y,z,t, + channel

    # We know the tank's dimensions, and the maximum possible travel time an unscattered photon can take
    # (from one corner to the other)
    # So assuming the first hit time (or the mean) is more than this travel time's "distance" away from the furthest
    # hit times, we should remove those later hits times since they were probably from later interactions/scattering

    # find the mean of the first 4 hits - there may be some isolated initial hits that are not a representation of the "first"
    # interaction hits
    
    av_temp = [];
    for i in range(len(hitT[event])):
        av_temp.append(hitT[event][i])      # in order to not mess up the initial hitT[i] array    
    av_temp.sort()

    first_time = sum(av_temp[0:4])/4
    
   
    max_travel_time = (np.sqrt(height_detector**2 + ((2*radius_detector)**2))/c)*(1e9)

    # also impose that there are no repeat hits on the same PMT's (second if statement)
    # (if PMT i was hit at time t, then hit later at time t' -- the hits had to be from a later
    # interaction.)
    for i in range(len(hitT[event])):
        if (first_time - max_travel_time) < hitT[event][i] < (first_time + max_travel_time):
            # in principle, this also rejects a hit (or two) in the cluster that may have dramatically
            # preceeded the others.
            
            if Channel[event][i] not in filtered_hits[4]:    # discard multiple hits on the same PMT
                       
                filtered_hits[3].append(hitT[event][i])

                filtered_hits[0].append(hitX[event][i])      # to keep the indexing consistent
                filtered_hits[1].append(hitY[event][i])
                filtered_hits[2].append(hitZ[event][i])

                filtered_hits[4].append(Channel[event][i])   # channel info
                
                
    isolated_counter = len(hitT[event]) - len(filtered_hits[0])

    print('removed ', isolated_counter, ' isolated hits')
    
    # # # # # # # # # # # # # # # # # # #

    # hit reduction - apply a more stringent cut on hit times
    
    # testing every pair from N choose k combinations for the causality condition
    # turned out to be far too computationally expensive. We will perform a shortcut. 
    
    # we already removed the most isolated hits from the arrays, and instead of looping over
    # every possible pair, testing to see if they could be causally connected, we can instead do
    # the following:
    
    # assume the first four points that made up the "average" initial time (above) satisfies the condition. 
    # (assume those hits can from the same, starting interaction point -- probably need to re-check this),
    # you can loop through each later hit and test to see if each one of those hits are causally independent
    # from the initial 4. If any are not, remove them from the array
    
    a_t = [[], [], [], [], []];        # initial 4 hits (x,y,z,t,channel)
    
    min_filt = min(filtered_hits[3])   # probably some degeneracy
    
    for time in range(0, 50, 2):
        if len(a_t[0]) == 4:
                break
        # have to make multiple passes to grab the "next" min value after t0
        for i in range(len(filtered_hits[3])):
            if len(a_t[0]) == 4:
                break
            else:
                if filtered_hits[3][i] == min_filt:         # append the minimum time
                    if filtered_hits[4][i] not in a_t[4]:   # no multi-hit PMTs in here
                        a_t[3].append(filtered_hits[3][i])
                        a_t[0].append(filtered_hits[0][i])
                        a_t[1].append(filtered_hits[1][i])
                        a_t[2].append(filtered_hits[2][i])
                        a_t[4].append(filtered_hits[4][i]) 
                        
        min_filt += 2   # since time is digitized to 2ns, go to the next time (t + dt)
    
   
    reject_count = 0
    
    for i in range(len(filtered_hits[3])):    
        # if the hit is not part of the initial 4
        if filtered_hits[3][i] not in a_t[3]:
            
            reject = False
            
            # test the hit not in the initial hits vs those in the initial hits
            for oh in range(len(a_t[3])):
            
                t_i = filtered_hits[3][i]; t_j = a_t[3][oh]

                # distance between PMT i and j
                dX = np.sqrt((filtered_hits[0][i] - a_t[0][oh])**2 + \
                             (filtered_hits[1][i] - a_t[1][oh])**2 + \
                                 (filtered_hits[2][i] - a_t[2][oh])**2)

                cond = arrival_t_criteria(t_i,t_j,dX)  # returns true is condition is upheld

                if cond == False:                      # if criterion is violated, reject outside hit
                    reject = True
                    break                              # stop testing against other hits
                    
            if reject == False:        # if the condition was satisfied with all initial hits
                
                HIT[event][0].append(filtered_hits[0][i])
                HIT[event][1].append(filtered_hits[1][i])
                HIT[event][2].append(filtered_hits[2][i])
                HIT[event][3].append(filtered_hits[3][i])
                
            else:
                
                reject_count += 1
                
        else:       # if hit is part of the initial ones, include it
                
            HIT[event][0].append(filtered_hits[0][i])
            HIT[event][1].append(filtered_hits[1][i])
            HIT[event][2].append(filtered_hits[2][i])
            HIT[event][3].append(filtered_hits[3][i])
            
    print('rejected ', reject_count, ' additional hits')
    
    if len(hitT[event]) != (len(HIT[event][3]) + reject_count + isolated_counter):
        print('********************')
        print('Error in filterization/reject!!! Length of arrays do not match')
        print('********************')
    
    print(len(HIT[event][3]), 'final hits:')
    print(HIT[event][3], '\n')




# ----------------------------------------
# Part 3 - MCMC Reconstruction using emcee

print('\nPerforming reconstruction...\n')

print('# # # # # # # # # # # # # # # # # # # # # # # # #')
print('For each event, MCMC will run an initial burn')
print('then run a longer sampler burn (two progress bars in total)')
print('After the runs are complete, the walker distributions')
print('will be fit, and the most likely vertex position will')
print('be found. Errors will be calculated based on the truth vertex info')
print('\n')

# # # # # # # # # # # # # # # # # #
# Time residual and log likelihood

# read in PDF information from fit_PDF_residual.py --> fits non-central student's t PDF to hit-timing residual data
PDF_params = np.loadtxt('PDF.dat', dtype = float, delimiter = None, skiprows = 1)

# scipy.stats nct function:  nct(df, nc, loc, scale)   (.dat file columns are in this order)
nct_df = PDF_params.T[0]
nct_nc = PDF_params.T[1]
nct_loc = PDF_params.T[2]
nct_scale = PDF_params.T[3]

def log_prob(Vertex, X, Y, Z, T):   # first arg is the N-dim np array (position of a single "walker")
                                    # following args are constant every time the function is called (hit info)
    temp_array = []
    
    # we can actually confine the walkers to a finite volume of the parameter space
    # if the vertex position is outside the tank (sufficiently far), return -infinity
    # - inf outside of the volume corresponds to the logarithm of 0 prior probability

    # Currently, only positional priors are used. Testing on a few events showed adding a prior for time
    # didn't dramatically change the results. Maybe add a smart one in the future (~3 meters in ct is ~14 ns travel time in water)
    
    if ((np.abs(Vertex[0]) < 3) and (np.abs(Vertex[1]) < 3) and (np.abs(Vertex[2]) < 3)):
    
        for ik in range(len(T)):
            # for each PMT hit in an event, find the hit time residual WRT each reconstructed vertex in the event
            tof_i = np.abs(np.sqrt( (X[ik] - Vertex[0])**2 + (Y[ik] - Vertex[1])**2 + (Z[ik] - Vertex[2])**2 ))/c

            d_t = (T[ik]/(1e9) - tof_i - Vertex[3]/c)*(1e9)   # time residual | convert to ns for the PDF

            pval = nct.pdf(d_t, df=nct_df, nc=nct_nc, loc=nct_loc, scale=nct_scale)    # associated probability density function
            
            temp_array.append(np.log(pval))                   # log probability

            
        sum_temp = sum(temp_array)                            # log likelihood
    
    else:      # return -infinity if vertex "guess" is way outside volume of detector
        
        sum_temp = -np.inf
    
    return sum_temp               

# # # # # # # # # # # # # # # # # # # # # 
# finding vertex solution in 4-dimensions
ndim = 4                # (x,y,z,ct)

print('Number of Walkers =', nwalkers)
print(burn_in_steps, 'burn-in steps |', n_steps, 'production-run steps\n')

# instead of creating a ~5-10 MB file for each event, we can perform the full
# reconstruction (maximization fitting), then append a general .dat file containing
# the reco positions, the truth information, and the errors.

# We want to be able to "save" our progress. It takes a long time to run many events, and while
# I haven't encountered any errors, if you want/need to stop the code for whatever reason,
# you want to be able to close the .dat file that is continually being written to even if you quit.
# "try" and "finally" will allow us to close the file no matter what stops the code.

# final reconstructed vertex information to be saved
file1 = open('emcee_files/Mean_Acceptance_Fraction.dat', "w")
file2 = open('emcee_files/r_error_' + str(a_start) + '_' + str(a_end) + '.dat', "w")
file3 = open('emcee_files/reco_pos_' + str(a_start) + '_' + str(a_end) + '.dat', "w")
file4 = open('emcee_files/reco_errors_' + str(a_start) + '_' + str(a_end) + '.dat', "w")
file5 = open('emcee_files/truth_pos_' + str(a_start) + '_' + str(a_end) + '.dat', "w")


try:
    
    for event in range(a_start,a_end):

        print('########## Event ' + str(event) + ' ##########')

        truth_v = [vtX[event], vtY[event], vtZ[event], vtT[event]]    # true vertex

        file2.write('Number eventNumber clusterNumber total_positional_error[cm]\n')
        file3.write('Number eventNumber clusterNumber X_reco_pos[cm] Y_reco_pos[cm] Z_reco_pos[cm] T_reco_time[ns]\n')
        file4.write('Number eventNumber clusterNumber X_error[cm] Y_error[cm] Z_error[cm] T_error[ns]\n')

        file5.write('Number eventNumber clusterNumber truth_vtX[cm] truth_vtY[cm] truth_vtZ[cm] truth_vtT[ns]\n')
        file5.write(str(event+a_start) + ' ' + str(clustereventNumber[event]) + ' ' + str(clusterNumber[event]) + ' ' + \
                    str(truth_v[0]*100) + ' ' + str(truth_v[1]*100) + ' ' + str(truth_v[2]*100) + ' ' + str(truth_v[3]) + '\n')
                                                

        p0 = np.random.rand(nwalkers, ndim)   # initial guess

        for i in range(nwalkers):

            p0[i][0] = (p0[i][0]*2*radius_detector) - radius_detector  # -radius to +radius
            p0[i][1] = (p0[i][1]*height_detector) - height_detector/2  # -height to +height
            p0[i][2] = (p0[i][2]*2*radius_detector) - radius_detector  # -radius to +radius
	
            p0[i][3] = ((p0[i][3]*2 - 1 + (clusterTime[event] - 10))/(1e9))*c  # 2ns spread centered around ~10ns before the mean cluster time
            # narrow initial guess around 10 ns before the mean cluster hit time (before filtering)
            # MC says that the average hit time for low E electrons is ~10 ns after emission time t0 (from 2.5 to 30 MeV)

        # Get the ensemble sampler object
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args = [HIT[event][0], HIT[event][1], HIT[event][2], HIT[event][3]])

        # Perform burn-in steps in the MCMC chain to let the walkers explore the parameter space a bit
        # and get settled into the maximum of the density.

        state = sampler.run_mcmc(p0, burn_in_steps, progress = prg_bar)   # save final position of the walkers (after N steps)
        sampler.reset()   # call to this method clears all of the important bookkeeping parameters
                          # in the sampler so we get a fresh start.

        # Now, we can do our production run of N steps
        sampler.run_mcmc(state, n_steps, progress = prg_bar)

        # Mean acceptance fraction of the run
        file1.write("Event " + str(event) + 
                    " Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)) + '\n')

        samples = sampler.get_chain(flat = True)

        # MCMC calculates the vertex w/ time values of [ct]. Must convert back into [ns] for output

        fig_corner = corner.corner(samples, truth_color='#4e4dc9', smooth = 0.9, color = 'k',
                                   labels = ['x [m]', 'y [m]', 'z [m]', 'ct'],
                                   truths = truth_v)
        if save_picture == True:
            if event % save_frequency == 0:         # only save some pictures (good indicator of progress)
                fig_corner.savefig("emcee_plots/" + str(event) + "_corner.png",
                               bbox_inches='tight', dpi=300, pad_inches=.3,facecolor = 'w')

        plt.close()   # close current figure so as not to consume too much memory

        
        # -------------------------------------------------------------- #
        # Part 4 - Fit walker distributions to extract most likely vertex
        
        reco_pos = []; reco_error = []
        
        for j in range(4):       # calculate reconstructed position and error for each dimension

            fig, ax = plt.subplots()

            data = pd.Series(samples[:,j])  # transform data into panda series

            # Generate Kernel Density Estimate plot using Gaussian kernels
            data.plot.hist(density=True, ax=ax)
            data.plot.kde(ax=ax)

            # find max value of the fit
            data_pt = ax.lines[0].get_xydata()
            max_val = data_pt[np.where(data_pt[:, 1] == max(data_pt[:, 1]))]
            plt.close()   # close the plot so we dont leave N plots open when running

            # space coordinates
            if j != 3:    # don't convert time units
                reco_position = max_val[0][0]*100   # [cm]
                error = (max_val[0][0] - truth_v[j])*100   # calculate error
            # time coordinates
            else:
                reco_position = (max_val[0][0]/c)*(1e9)  # [ns]
                error = (max_val[0][0]/c)*(1e9) - truth_v[j]
                
            reco_error.append(error)
            reco_pos.append(reco_position)
               
        # find the total reconstructed error
        total_error = np.sqrt(reco_error[0]**2 + reco_error[1]**2 + reco_error[2]**2)
        
        # export reconstructed vertex information
        file2.write(str(event+a_start) + ' ' + str(clustereventNumber[event]) + ' ' + str(clusterNumber[event]) + ' ' + str(total_error) + '\n')
        file3.write(str(event+a_start) + ' ' + str(clustereventNumber[event]) + ' ' + str(clusterNumber[event]) + ' ' + \
                    str(reco_pos[0]) + ' ' + str(reco_pos[1]) + ' ' + str(reco_pos[2]) + ' ' + str(reco_pos[3]) + '\n')
        file4.write(str(event+a_start) + ' ' + str(clustereventNumber[event]) + ' ' + str(clusterNumber[event]) + ' ' + \
                    str(reco_error[0]) + ' ' + str(reco_error[1]) + ' ' + str(reco_error[2]) + ' ' + str(reco_error[3]) + '\n')
        
        print('Total Reconstructed Error: ', round(total_error,2), ' cm')

finally:            # execute even with errors (or closing the program)
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()

print('\ndone\n')    
