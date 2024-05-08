################################################################
# MCMC-based Python Reconstruction for low energy ANNIE Events #
# ------------------------------------------------------------ #
#   Using the hit time residuals in the tank (PMT only) and a  #
#    BONSAI likelihood function, perform an affine-invariant   #
#      Markov-Chain Monte Carlo (MCMC) ensembler sampler to    #
#   reconstruct the most likely vertex position in (x,y,z,ct). #
#       MCMC carried out through the python module, emcee.     #
# ------------------------------------------------------------ #
#            Author: Steven Doran   Date: May 2024             #
################################################################

import emcee
import corner
import sys, os
import warnings
import matplotlib
import numpy as np
import pandas as pd
from tqdm import trange
import uproot
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import genhyperbolic
from scipy.stats._continuous_distns import _distn_names


''' Things to check before you run this code:
    1. Is there a root file in /WCSim_data?
    2. Did you run fit_PDF_residual.py to generate a PDF fit to the time-residual data?
'''

######################################################
# ------ Parameterization - Requires User Input ---- #
######################################################

# 1. ---------------
file_name = ['skyshine_10k.ntuple.root']

# 2. ---------------
# Number of events/clusters we are performing the reconstruction on
start_event = int(sys.argv[1])
final_event = int(sys.argv[2]) 

# 3. ---------------
# emcee ensemble parameters
nwalkers = 1000         # number of walkers
burn_in_steps = 100      # number of burn-in steps to "feel out" the probability space
n_steps = 200            # number of steps for walkers in the primary run
prg_bar = False          # enable/disable progress bar. For grid submission, set = False

save_picture = False    # specify whether you want to save the .png corner plots

######################################################

class Cluster:
    def __init__(self,file_number,event_number,cluster_number,cluster_hits,hitX,hitY,hitZ,hitT,hitPE,hitID,
                 vtX,vtY,vtZ,vtTime):
        self.file_number = file_number
        self.event_number = event_number
        self.cluster_number = cluster_number
        self.cluster_hits = cluster_hits
        self.hitX = hitX
        self.hitY = hitY
        self.hitZ = hitZ
        self.hitT = hitT
        self.hitPE = hitPE
        self.hitID = hitID
        self.vtX = vtX
        self.vtY = vtY
        self.vtZ = vtZ
        self.vtTime = vtTime
        
clusters = []
        
c = 299792458  # [m/s]
c = c/(4/3)    # refractive index of water

# -------------------------------------
# Part 1 - Load in Data

print('\nLoading data...\n')

# Import and read PMT timing 
df = pd.read_csv('PMT_Timing.csv')
reliability = []; ch_num = []; res = []
for i in range(len(df['Channel'])):       # loop over PMTs
    ch_num.append(df['Channel'][i])
    reliability.append(df['notes'][i])
    res.append(df['offset_std(ns)'][i])


for files in range(len(file_name)):

    file = uproot.open(file_name[files])

    T = file['phaseIITankClusterTree']
    CEN = T['eventNumber'].array()
    CN = T['clusterNumber'].array()
    CH = T['clusterHits'].array()
    
    # Only grab hits with "reliable" hit timing
    X1 = T['hitX'].array()
    Y1 = T['hitY'].array()
    Z1 = T['hitZ'].array()
    T1 = T['hitT'].array()
    PE1 = T['hitPE'].array()
    ID1 = T['hitDetID'].array()
    tempX = [[] for i in range(len(X1))]; tempY = [[] for i in range(len(Y1))]; tempZ = [[] for i in range(len(Z1))]
    tempT = [[] for i in range(len(T1))]; tempPE = [[] for i in range(len(PE1))]; tempID = [[] for i in range(len(ID1))]
    for i in range(len(ID1)):
        for j in range(len(ID1[i])):
            if ID1[i][j] in ch_num:
                indy = ch_num.index(ID1[i][j])
                if reliability[indy] == 'ok':
                    tempX[i].append(X1[i][j])
                    tempY[i].append(Y1[i][j])
                    tempZ[i].append(Z1[i][j])
                    tempT[i].append(T1[i][j])
                    tempPE[i].append(PE1[i][j])
                    tempID[i].append(ID1[i][j])

    # truth information
    truth = file['phaseIITriggerTree']
    en = truth['eventNumber'].array()
    vx = truth['trueNeutCapVtxX'].array()         # [cm]
    vy = truth['trueNeutCapVtxY'].array()
    vz = truth['trueNeutCapVtxZ'].array()
    vt = truth['trueNeutCapTime'].array()

    for i in range(len(CEN)):
        event = Cluster(file_number=files,event_number=CEN[i],cluster_number=CN[i],cluster_hits=CH[i],
                        hitX=tempX[i],hitY=tempY[i],hitZ=tempZ[i],hitT=tempT[i],hitPE=tempPE[i],hitID=tempID[i],
                        vtX=vx[CEN[i]][0]/100,vtY=(vy[CEN[i]][0]/100 + 0.1446),vtZ=(vz[CEN[i]][0]/100 - 1.681),vtTime=vt[CEN[i]][0]
        )
        clusters.append(event)


N_clusters = len(clusters)
print('\nData Loaded\n')
print('#################')
print('\nNumber of Clusters = ', N_clusters)
print('\nRunning ', (final_event - start_event + 1), ' cluster(s) (start = cluster ', start_event, ', final = cluster ', final_event, ')') 

# -------------------------------------
# Part 2 - Filter hits

print('\nFiltering hits...\n')

# rejection criteria based on causality
def arrival_t_criteria(ti, tj, dX):
    delta_tij = np.abs(ti - tj)
    tof = ((np.abs(dX)/c)*1e9)
    if delta_tij < tof:
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

not_enough = [0 for i in range(N_clusters)]

for event in range(a_start,a_end+1):
    
    print('########## Cluster ', event, ' ##########')
    
    print(clusters[event].cluster_hits, 'initial hits in the cluster')
    print(len(clusters[event].hitT), 'hits on PMTs with reliable timing')

    if len(clusters[event].hitT) < 4:
        print('\n**** NOT ENOUGH HITS FOR RECONSTRUCTION - VERTEX RECO WILL SKIP THIS EVENT ****\n')
        not_enough[event] = 1
        continue

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # remove isolated hits 

    filtered_hits = [[], [], [], [], []]   # x,y,z,t, + channel
    
    av_temp = [];
    for i in range(len(clusters[event].hitT)):
        av_temp.append(clusters[event].hitT[i])      # in order to not mess up the initial hitT[i] array    
    av_temp.sort()

    first_time = sum(av_temp[0:4])/4
    
    max_travel_time = 10

    for i in range(len(clusters[event].hitT)):

        # in principle, this condition also rejects a hit (or two) in the cluster that may have dramatically preceeded the others
        if (first_time - max_travel_time) < clusters[event].hitT[i] < (first_time + max_travel_time):

            if clusters[event].hitID[i] not in filtered_hits[4]:    # discard multiple hits on the same PMT
                       
                filtered_hits[3].append(clusters[event].hitT[i])
                filtered_hits[0].append(clusters[event].hitX[i])
                filtered_hits[1].append(clusters[event].hitY[i])
                filtered_hits[2].append(clusters[event].hitZ[i])
                filtered_hits[4].append(clusters[event].hitID[i])
                
                
    isolated_counter = len(clusters[event].hitT) - len(filtered_hits[0])
    print('removed ', isolated_counter, ' isolated hits')
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # hit reduction - apply a more stringent cut on hit times

    # we already removed the most isolated hits from the arrays, and instead of looping over
    # every possible pair, testing to see if they could be causally connected, we can instead do
    # the following:
    
    # assume the first four points that made up the "average" initial time (above) satisfies the condition. 
    # (assume those hits can from the same, starting interaction point)
    # Loop through each later hit and test to see if each one of those hits are causally independent
    # from the initial 4. If any are not, remove them from the array.
    
    a_t = [[], [], [], [], []];        # initial 4 hits (x,y,z,t,channel)
    
    min_filt = min(filtered_hits[3])
    
    fil_temp = []
    for i in range(len(filtered_hits[3])):
        fil_temp.append(filtered_hits[3][i])
    fil_temp.sort()
    
    for time in fil_temp:
        if len(a_t[0]) == 4:          # quit after we find the first 4
            break
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
                        
        min_filt = time   # next hit time
        
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

                cond = arrival_t_criteria(t_i,t_j,dX)     # returns true is condition is upheld

                if cond == False:                         # if criterion is violated, reject outside hit
                    reject = True
                    break                                 # stop testing against other hits
                    
            if reject == False:                           # if the condition was satisfied with all initial hits
                HIT[event][0].append(filtered_hits[0][i])
                HIT[event][1].append(filtered_hits[1][i])
                HIT[event][2].append(filtered_hits[2][i])
                HIT[event][3].append(filtered_hits[3][i])
                
            else:
                reject_count += 1
                
        else:                                             # if hit is part of the initial ones, include it
            HIT[event][0].append(filtered_hits[0][i])     # if there are only 4 hits from the PMT reliaiblity cut, all 4 will be included
            HIT[event][1].append(filtered_hits[1][i])
            HIT[event][2].append(filtered_hits[2][i])
            HIT[event][3].append(filtered_hits[3][i])
            
    print('rejected ', reject_count, ' additional hits')
    
    if len(clusters[event].hitT) != (len(HIT[event][3]) + reject_count + isolated_counter):
        print('********************')
        print('Error in filterization/reject!!! Length of arrays do not match')
        print('********************')
    
    if len(HIT[event][3]) < 4:
        print('\n**** NOT ENOUGH HITS FOR RECONSTRUCTION (ONLY ' + str(len(HIT[event][3])) + ' FINAL HITS) - VERTEX RECO WILL SKIP THIS EVENT ****\n')
        not_enough[event] = 1
        continue
    
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

# read in PDF information from fit_PDF_residual.py
PDF_params = np.loadtxt('PDF.dat', dtype = float, delimiter = None, skiprows = 1)
PDF_header = np.loadtxt('PDF.dat', dtype = str, delimiter = None, max_rows = 1)
print('MCMC will sample from the following PDF distribution: ', PDF_header, '\n')
                
# scipy.stats generalized hyperbolic function:  genhyperbolic(p, a, b, loc, scale)
p_1 = PDF_params.T[0]; a_1 = PDF_params.T[1]; b_1 = PDF_params.T[2]; loc_1 = PDF_params.T[3]; scale_1 = PDF_params.T[4]


def log_prob(Vertex, X, Y, Z, T):   # first arg is the N-dim np array (position of a single "walker")
                                    # following args are constant every time the function is called (hit info)
    temp_array = []
    
    # priors:
    #         - event vertex will be somewhere in the ANNIE water volume
    #         - emission time must be before the first hit time
    #         - emission time can't be more than ~16ns before first hit (maximum propagation time from one corner of PMT rack to another)
    if (np.sqrt(Vertex[0]**2 + Vertex[2]**2) < 1.524) and (np.abs(Vertex[1]) < 1.98) and ((min(T)-16) < Vertex[3] < min(T)):

        for ik in range(len(T)):
            # for each PMT hit in an event, find the hit time residual WRT each reconstructed vertex in the event
            tof_i = np.abs(np.sqrt( (X[ik] - Vertex[0])**2 + (Y[ik] - Vertex[1])**2 + (Z[ik] - Vertex[2])**2 ))/c

            d_t = (T[ik]/(1e9) - tof_i - Vertex[3]/(1e9))*(1e9)               # time residual | convert to ns for the PDF

            pval = genhyperbolic.pdf(d_t, p=p_1, a=a_1, b=b_1, loc=loc_1, scale=scale_1)    # associated probability density function
            
            temp_array.append(np.log(pval))                               # log probability

            
        sum_temp = sum(temp_array)                                        # log likelihood
    
    else:      # return -infinity if vertex "guess" is outside volume of detector
        
        sum_temp = -np.inf
    
    return sum_temp
        

# # # # # # # # # # # # # # # # # # # # # 
# finding vertex solution in 4-dimensions
ndim = 4                # (x,y,z,ct)

print('Number of Walkers =', nwalkers)
print(burn_in_steps, 'burn-in steps |', n_steps, 'production-run steps\n\n')

# final reconstructed vertex information to be saved
root_str = 'output/vtx_reco_' + str(a_start) + '_' + str(a_end) + '.ntuple.root'
os.system('rm ' + root_str)        # in case there is some pre-existing file
file = uproot.create(root_str)

evs = [[], []]             # event number, cluster number
vtxs = [[], [], [], []]    # x, y, z [cm], t [ns]   - truth vertices
er = [[], [], [], [], []]  # x, y, z [cm], t [ns], total error [cm]   - errors
r_pos = [[], [], [], []]   # x, y, z [cm], t [ns]   - reconstructed positions
n_hits = []                # final number of filtered hits used in reconstruction


try:
    
    for event in range(a_start,a_end+1):

        if not_enough[event] == 1:
            print('\n**** NOT ENOUGH HITS FOR RECONSTRUCTION - VERTEX RECO WILL SKIP THIS EVENT ****\n')
            continue

        print('########## Event ' + str(event) + ' ##########')
        print('vertex: (x,y,z,t):', clusters[event].vtX, ' ', clusters[event].vtY, ' ', clusters[event].vtZ, ' ', clusters[event].vtTime)

        truth_v = [clusters[event].vtX, clusters[event].vtY, clusters[event].vtZ, clusters[event].vtTime]        

        p0 = np.random.rand(nwalkers, ndim)   # initial guess
        
        for i in range(nwalkers):
            
            # (1.118573625, 3.0338736500000003)  PMT rack radius and height [m]   /   (1.5, 1.95) Tank radius and halfz height [m]

            # initial distribution is a rectangular prism
            p0[i][0] = (p0[i][0]*3) - 1.5  # water tank -radius to +radius
            p0[i][1] = (p0[i][1]*4) - 2    # water tank -height to +height
            p0[i][2] = (p0[i][2]*3) - 1.5  # water tank -radius to +radius
	
            # time is put into units of [c*seconds]
            p0[i][3] = (p0[i][3]*10 - 5 + (min(HIT[event][3]) - 5))  # 10ns spread centered around ~10ns before the mean cluster time

        # Get the ensemble sampler object
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args = [HIT[event][0], HIT[event][1], HIT[event][2], HIT[event][3]])

        # Perform burn-in steps in the MCMC chain to let the walkers explore the parameter space a bit
        # and get settled into the maximum of the density.

        state = sampler.run_mcmc(p0, burn_in_steps, progress = prg_bar)   # save final position of the walkers (after N steps)
        sampler.reset()   # call to this method clears all of the important bookkeeping parameters
                          # in the sampler so we get a fresh start.

        # Now, we can do our production run of N steps
        sampler.run_mcmc(state, n_steps, progress = prg_bar)

        samples = sampler.get_chain(flat = True)

        fig_corner = corner.corner(samples, truth_color='#4e4dc9', smooth = 0.9, color = 'k',
                                   labels = ['x [m]', 'y [m]', 'z [m]', 't [ns]'],
                                   truths = truth_v)
        if save_picture == True:
            fig_corner.savefig("emcee_plots/" + str(event) + "_corner.png",
                           bbox_inches='tight', dpi=300, pad_inches=.3,facecolor = 'w')

        plt.close()   # close current figure so as not to consume too much memory

        
        # -------------------------------------------------------------- #
        # Part 4 - Fit walker distributions to extract most likely vertex
        
        reco_error = []
        
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
                error = (max_val[0][0] - truth_v[j])*100   # calculate error [cm]
                vtxs[j].append(truth_v[j]*100)
            
            # time coordinates - MCMC calculates the vertex w/ time values of [ct]. Must convert back into [ns] for output
            else:
                reco_position = max_val[0][0]    # [ns]
                error = (max_val[0][0] - truth_v[j])
                print('Reconstructed Emission Time Error: ', round(error,2), ' ns')
                vtxs[j].append(truth_v[j]) 
                
            reco_error.append(error)
            r_pos[j].append(reco_position)
            er[j].append(error)
               
        # find the total reconstructed error
        total_error = np.sqrt(reco_error[0]**2 + reco_error[1]**2 + reco_error[2]**2)
        er[4].append(total_error)
        
        print('Total Reconstructed Error: ', round(total_error,2), ' cm\n')

        n_hits.append(len(HIT[event][0]))
        evs[0].append(clusters[event].event_number); evs[1].append(clusters[event].cluster_number)


finally:            # execute even with errors (or closing the program)
    
    tree_data = {
        "eventNumber": np.array(evs[0]),
        "clusterNumber": np.array(evs[1]),
        "final_Nhits": np.array(n_hits),
        "truth_x": np.array(vtxs[0]),
        "truth_y": np.array(vtxs[1]),
        "truth_z": np.array(vtxs[2]),
        "truth_t": np.array(vtxs[3]),
        "reco_pos_x": np.array(r_pos[0]),
        "reco_pos_y": np.array(r_pos[1]),
        "reco_pos_z": np.array(r_pos[2]),
        "reco_pos_t": np.array(r_pos[3]),
        "error_x": np.array(er[0]),
        "error_y": np.array(er[1]),
        "error_z": np.array(er[2]),
        "error_t": np.array(er[3]),
        "total_error": np.array(er[4]),
    }    

    file["Vertex_Reconstruction"] = tree_data
    file.close()
        

print('\ndone\n')
