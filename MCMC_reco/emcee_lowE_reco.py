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


''' Things to check before you run this code:
    1. Did you dump the root tree information into .dat files using the .C scripts?
    2. Did you run fit_PDF_residual.py to generate a PDF fit to the time-residual data? '''


''' ###################################################### '''
''' # ------ Parameterization - Requires User Input ---- # '''
''' ###################################################### '''

# 1. ---------------
# path extension names for .dat files containing root tree entries
position = 'swarm_15MeV'
folder = 'WCSim_Data/15MeV/'

# 2. ---------------
# Display the detector Geometry with truth vertices? (Will display external window)
DisplayGeometry = False

# 3. ---------------
# Number of events we are performing the reconstruction on
# *** (It is recommended to run this code in chunks (maybe 100 events at a time)) ***

run_all = False               # will run all events

# pass the starting and ending event number as arguments to the script
start_event = int(sys.argv[1])     # if run_all == True, these won't matter
final_event = int(sys.argv[2])     # can pass this as: N_events (total number of events; if running like 100:N_events)

# 4. ---------------
# emcee parameters
nwalkers = 1000         # number of walkers
burn_in_steps = 50      # number of burn-in steps to "feel out" the probability space
n_steps = 100           # number of steps for walkers in the primary run
prg_bar = False         # enable/disable progress bar. For grid submission, set = False

save_picture = False    # specify whether you want to save the .png corner plots
save_first_ten = False  # often useful for diagnostics, see if the reco algorithm is working before doing more events
save_frequency = 10     # can be used as code update, if save_picture is enabled, the code will save every (save_frequency)-th picture
                        # set to 1 if you want all pictures saved 

''' ###################################################### '''

# -------------------------------------
# Part 1 - Load in Data

print('Loading data...')

# # # # # # # # # # # # # # #
# Event-level information
path_charge = folder + 'charge_event_electron_' + position + '.dat'
event_header = np.loadtxt(path_charge, dtype = str, delimiter = None, max_rows = 1)
event_data = np.loadtxt(path_charge, dtype = float, delimiter = None, skiprows = 1)

clustereventNumber = event_data.T[0]
clusterCharge = event_data.T[1]
clusterPE = event_data.T[2]
clusterMaxPE = event_data.T[3]
clusterChargeBalance = event_data.T[4]
clusterHits = event_data.T[5]

N_events = len(clustereventNumber)

# Now load in the hits-level information
path_hits = folder + 'cluster_hits_electron_' + position + '.dat'
hits_header = np.loadtxt(path_hits, dtype = str, delimiter = None, max_rows = 1)
hits_data = np.loadtxt(path_hits, dtype = float, delimiter = None, skiprows = 1)

Channel = [[] for i in range(N_events)]
hitT = [[] for i in range(N_events)]      # The x,y,z is adjusted correctly in the ToolChain for this Event Display
hitX = [[] for i in range(N_events)]
hitY = [[] for i in range(N_events)]
hitZ = [[] for i in range(N_events)]
hitQ = [[] for i in range(N_events)]
hitPE = [[] for i in range(N_events)]

count = 0
for j in range(len(hits_data.T[0])):    # loop over all hits (N events x M hits per event)
    if (j == 0):
        Channel[count].append(hits_data.T[1][j])
        hitT[count].append(hits_data.T[2][j])      # the hitX, hitY, hitZ contain the position of the
        hitX[count].append(hits_data.T[3][j])      # PMTs that were hit
        hitY[count].append(hits_data.T[4][j])      
        hitZ[count].append(hits_data.T[5][j])
        hitQ[count].append(hits_data.T[6][j])
        hitPE[count].append(hits_data.T[7][j])
        
    elif (j != 0):
        if hits_data.T[0][j] == hits_data.T[0][j-1]:
            Channel[count].append(hits_data.T[1][j])
            hitT[count].append(hits_data.T[2][j])
            hitX[count].append(hits_data.T[3][j])
            hitY[count].append(hits_data.T[4][j])
            hitZ[count].append(hits_data.T[5][j])
            hitQ[count].append(hits_data.T[6][j])
            hitPE[count].append(hits_data.T[7][j])
        else:
            count = count + 1
            Channel[count].append(hits_data.T[1][j])
            hitT[count].append(hits_data.T[2][j])
            hitX[count].append(hits_data.T[3][j])
            hitY[count].append(hits_data.T[4][j])
            hitZ[count].append(hits_data.T[5][j])
            hitQ[count].append(hits_data.T[6][j])
            hitPE[count].append(hits_data.T[7][j])
            
# Load in the MC Truth information (there will be more events than clusters -- need to sort and assign them)
path_truth = folder + 'mctruth_electron_' + position + '.dat'
truth_header = np.loadtxt(path_truth, dtype = str, delimiter = None, max_rows = 1)
truth_data = np.loadtxt(path_truth, dtype = float, delimiter = None, skiprows = 1)

eventNumber = truth_data.T[0]  # Event Number
vtX = truth_data.T[1]          # {vertex information   
vtY = truth_data.T[2]          # ..
vtZ = truth_data.T[3]          # }
vtT = truth_data.T[4]           # "vertex time" i.e. initial time
dirX = truth_data.T[5]         # {direction vectors of primary particle
dirY = truth_data.T[6]         # ..
dirZ = truth_data.T[7]         # }
Energy = truth_data.T[8]       # initial energy of the primary particle
Track_Length = truth_data.T[9] # track length of the primary particle in water (distance from start point to stop point or the
                               # distance from the start vertex to a tank wall (if the particle exited))

# sort events that dont have an associated cluster event number
vtX = [vtX[int(x)]/100 for x in eventNumber if x in clustereventNumber]
vtY = [vtY[int(x)]/100 for x in eventNumber if x in clustereventNumber]
vtZ = [vtZ[int(x)]/100 for x in eventNumber if x in clustereventNumber]
# divide by 100     ^  to convert vertex units [cm] into [m]
vtT = [vtT[int(x)] for x in eventNumber if x in clustereventNumber]
dirX = [dirX[int(x)] for x in eventNumber if x in clustereventNumber]
dirY = [dirY[int(x)] for x in eventNumber if x in clustereventNumber]
dirZ = [dirZ[int(x)] for x in eventNumber if x in clustereventNumber]
Energy = [Energy[int(x)] for x in eventNumber if x in clustereventNumber]
Track_Length = [Track_Length[int(x)] for x in eventNumber if x in clustereventNumber]

#################################################################################
# We can also create some custom arrays

# Charge and hits
Sum_PE = [[] for i in range(N_events)]       # summed P.E. on each PMT
hits_per_PMT = [[] for i in range(N_events)] # number of hits on each PMT
unique_PMTs = [[] for i in range(N_events)]  # unique hit PMTs
for i in range(N_events):
    u, c = np.unique(Channel[i], return_counts=True)
    unique_PMTs[i].append(u.tolist())        # prevents this list from becoming a numpy array
    hits_per_PMT[i].append(c.tolist())       # (only included because we call the method index() later
                                             # on, which only works on lists)
for i in range(N_events):
    for j in range(len(unique_PMTs[i][0])):
        pe = 0.
        for k in range(len(Channel[i])):
            if unique_PMTs[i][0][j] == Channel[i][k]:
                pe = pe + hitPE[i][k]
        Sum_PE[i].append(pe)
        
# There exists a small descrepancy between clusterMaxPE and the summed Max PE per PMT
# The former clusterMaxPE is defined as the maximum PE of a given hit. The latter accounts
# for multiple hits on the same PMT, so it is the maximum summed charge over all PMTs. The
# difference over the entirity of the events is small though. 
        
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# MC Truth Vertex and direction information
origin = np.zeros([N_events,3])
dir_vector = np.zeros([N_events,3])
for i in range(N_events):
    origin[i][0] = vtZ[i]; dir_vector[i][0] = dirZ[i]
    origin[i][1] = vtX[i]; dir_vector[i][1] = dirX[i]
    origin[i][2] = vtY[i]; dir_vector[i][2] = dirY[i]

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# We can also define whether an event triggered an extended readout:
# (events with max p.e. > 7 for a given PMT)
ext_readout_trig = []
for i in range(N_events):
    if clusterMaxPE[i] > 7.:
        ext_readout_trig.append(True)
    else:
        ext_readout_trig.append(False)
        
# # # # # # # # # # # # # # # # # # # # # # # # # # #
### Load in and Construct the Detector Geometry ###

# Not necessary, but helpeful for displaying vertex positions

# Read Geometry.csv file to get PMT location info
df = pandas.read_csv('FullTankPMTGeometry.csv')

channel_number = []; location = []; panel_number = []
x_position = []; y_position = []; z_position = []

# The LoadGeometry File does not share the same origin point as the WCSim output after ToolAnalysis.
# Need to adjust to ensure the (0,0,0) point (center of the tank) is the same.
# This offset is confirmed by plotting it without adjustments (see below, commented)

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
    

## Optional for viewing events ##
# # # # # # # # # # # # # # # # #
#  Displaying the Tank Geometry #
# # # # # # # # # # # # # # # # #
#%matplotlib

if DisplayGeometry == True:
    
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)   # zorder was not manual as of 3.5.0

    # Plot each PMT location
    for i in range(len(channel_number)):
        if i == 0:
            ax.scatter(z_position[i], x_position[i], y_position[i], s = 50, color = 'black', label = 'PMTs', zorder = 5)
        else:
            ax.scatter(z_position[i], x_position[i], y_position[i], s = 50, color = 'black', zorder = 5)

    # Plot verticies 
    for i in range(len(vtX)):
        if i == 0:
            ax.scatter(vtZ[i],vtX[i],vtY[i], s = 5, color = 'red', marker = '*', zorder = 1, label = 'Event Verticies')
        else:
            ax.scatter(vtZ[i],vtX[i],vtY[i], s = 5, color = 'red', marker = '*', zorder = 1)

    ax.set_aspect('auto')
    ax.set_xlabel('z [m]')   # re-adjust the axes, to reflect the real-life geometry of the tank
    ax.set_ylabel('x [m]')   # for example, y is really the height dimension of the ANNIE tank,
    ax.set_zlabel('y [m]')   # but if you plotted it normally with a RH axes, it would show tank on its side

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([max(z_position)-min(z_position), max(x_position)-min(x_position), max(y_position)-min(y_position)]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(max(z_position)+min(z_position))
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(max(x_position)+min(x_position))
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(max(y_position)+min(y_position))
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.title('ANNIE PMT Geometry with Event Verticies')
    plt.legend(shadow = True)
    #path = '../WCSim Simulation Results/Plots/ANNIE Tank with electron swarm.png'
    #plt.savefig(path,dpi=300, bbox_inches='tight', pad_inches=.3,facecolor = 'w')
    plt.show()

    print('PMT Geometry Loaded\n')

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
# Find the dimensions of the detector (radius, height, etc..)
height_detector = max(y_position) - min(y_position)
radius_detector = (max(z_position) - min(z_position))/2   

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

print('\nData Loaded\n')
print('#################')
print('\nN Events = ', N_events)

if run_all == False:
	print('\nRunning ', (start_event + final_event), ' event(s) (start = event ', start_event, ', final = event ', final_event, ')') 

else:
	print('\nRunning over all (', N_events, ') events')
 

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


HIT = [[] for i in range(N_events)]        # final, filtered, reduced hits list
for i in range(len(HIT)):
    for j in range(4):
        HIT[i].append([])


if run_all == False:
	a_start = start_event
	a_end = final_event
else:
	a_start = 0
	a_end = N_events

for event in range(a_start,a_end):          # (how many events are you running?)
    
    print('########## Event ', event, ' ##########')
    
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
file4 = open('emcee_files/rec_errors_' + str(a_start) + '_' + str(a_end) + '.dat', "w")


try:
    
    for event in range(a_start,a_end):

        print('########## Event ' + str(event) + ' ##########')

        truth_v = [vtX[event], vtY[event], vtZ[event], vtT[event]]    # true vertex

        p0 = np.random.rand(nwalkers, ndim)   # initial guess

        for i in range(nwalkers):

            p0[i][0] = (p0[i][0]*2*radius_detector) - radius_detector  # -radius to +radius
            p0[i][1] = (p0[i][1]*height_detector) - height_detector/2  # -height to +height
            p0[i][2] = (p0[i][2]*2*radius_detector) - radius_detector  # -radius to +radius

            p0[i][3] = ((p0[i][3] + 10 - (sum(HIT[event][3])/len(HIT[event][3])))/(1e9))*c  
            # narrow region around 10 ns before the mean hit time
            # MC says that the average hit time for low E electrons is ~10 ns after emission time t0, so this is probably okay

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

            if j != 3:    # don't convert time units
                reco_position = max_val[0][0]*100   # [cm]
                error = (max_val[0][0] - truth_v[j])*100   # calculate error
            else:
                reco_position = max_val[0][0]
                error = max_val[0][0] - truth_v[j]
                
            reco_error.append(error)
            reco_pos.append(reco_position)
               
        # find the total reconstructed error
        total_error = np.sqrt(reco_error[0]**2 + reco_error[1]**2 + reco_error[2]**2)
        
        # export reconstructed vertex information
        file2.write(str(total_error) + '\n')
        file3.write(str(reco_pos[0]) + ' ' + str(reco_pos[1]) + ' ' + str(reco_pos[2]) + ' ' + str(reco_pos[3]) + '\n')
        file4.write(str(reco_error[0]) + ' ' + str(reco_error[1]) + ' ' + str(reco_error[2]) + ' ' + str(reco_error[3]) + '\n')
        
        print('Total Reconstructed Error: ', total_error, ' cm')

finally:            # execute even with errors (or closing the program)
    file1.close()
    file2.close()
    file3.close()
    file4.close()

print('\ndone\n')    
