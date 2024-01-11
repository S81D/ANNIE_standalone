################################################################
# MCMC-based Python Reconstruction for low energy ANNIE Events #
# ------------------------------------------------------------ #
#   Using the hit time residuals in the tank (PMT only) and a  # 
#    BONSAI likelihood function, perform an affine-invariant   #
#      Markov-Chain Monte Carlo (MCMC) ensembler sampler to    #
#   reconstruct the most likely vertex position in (x,y,z,ct). #
#       MCMC carried out through the python module, emcee.     #
# ------------------------------------------------------------ #
#         Author: Steven Doran   Date: January 2024            #
################################################################

import emcee
import corner
import sys, os
import warnings
import matplotlib
import numpy as np
import pandas as pd
from tqdm import trange
import uproot3 as uproot
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import exponnorm
from scipy.stats._continuous_distns import _distn_names


''' Things to check before you run this code:
    1. Is there a root file in /WCSim_data?
    2. Did you run fit_PDF_residual.py to generate a PDF fit to the time-residual data?
'''

''' ###################################################### '''
''' # ------ Parameterization - Requires User Input ---- # '''
''' ###################################################### '''

# 1. ---------------
file_name = ['WCSim_Data/michel_1.ntuple.root']

# 2. ---------------
# Number of events/clusters we are performing the reconstruction on
start_event = int(sys.argv[1])
final_event = int(sys.argv[2]) 

# 3. ---------------
# emcee ensemble parameters
nwalkers = 1000         # number of walkers
burn_in_steps = 25      # number of burn-in steps to "feel out" the probability space
n_steps = 50            # number of steps for walkers in the primary run
prg_bar = True          # enable/disable progress bar. For grid submission, set = False

save_picture = False    # specify whether you want to save the .png corner plots

''' ###################################################### '''

class Cluster:
    def __init__(self,file_number,event_number,cluster_number,pe,time,Qb,n_hits,hitX,hitY,hitZ,hitT,hitPE,hitID,
                 vtX=None,vtY=None,vtZ=None,energy=None):
        self.file_number = file_number
        self.event_number = event_number
        self.cluster_number = cluster_number
        self.pe = pe
        self.time = time
        self.Qb = Qb
        self.n_hits = n_hits
        self.hitX = hitX
        self.hitY = hitY
        self.hitZ = hitZ
        self.hitT = hitT
        self.hitPE = hitPE
        self.hitID = hitID
        self.vtX = vtX
        self.vtY = vtY
        self.vtZ = vtZ
        self.energy = energy
        
clusters = []
        
c = 299792458  # [m/s]
c = c/(4/3)    # refractive index of water

# -------------------------------------
# Part 1 - Load in Data

print('\nLoading data...\n')

for files in range(len(file_name)):

    file = uproot.open(file_name[files])

    T = file['phaseIITankClusterTree']
    CEN = T['eventNumber'].array()
    CN = T['clusterNumber'].array()
    CPE = T['clusterPE'].array()
    CCB = T['clusterChargeBalance'].array()
    CH = T['clusterHits'].array()
    CT = T['clusterTime'].array()
    X1 = T['hitX'].array()
    Y1 = T['hitY'].array()
    Z1 = T['hitZ'].array()
    T1 = T['hitT'].array()
    PE1 = T['hitPE'].array()
    ID1 = T['hitDetID'].array()

    # truth information
    truth = file['phaseIITriggerTree']
    en = truth['eventNumber'].array()
    vx = truth['trueVtxX'].array()         # [cm]
    vy = truth['trueVtxY'].array()
    vz = truth['trueVtxZ'].array()
    ENG = truth['trueMuonEnergy'].array()  # [MeV]

    for i in range(len(CEN)):
        if CT[i] < 100 and CN[i] == 0:   # for MC, reject secondary and delayed clusters
            event = Cluster(file_number=files,event_number=CEN[i],cluster_number=CN[i],pe=CPE[i],time=CT[i],
                Qb=CCB[i],n_hits=CH[i],hitX=X1[i],hitY=Y1[i],hitZ=Z1[i],hitT=T1[i],hitPE=PE1[i],hitID=ID1[i],
                            vtX=vx[CEN[i]]/100,vtY=vy[CEN[i]]/100,vtZ=vz[CEN[i]]/100,energy=ENG[CEN[i]]
            )
            clusters.append(event)
        
N_clusters = len(clusters)
print('\nData Loaded\n')
print('#################')
print('\nNumber of Clusters = ', N_clusters)
print('\nRunning ', (final_event - start_event), ' cluster(s) (start = cluster ', start_event, ', final = cluster ', final_event, ')') 


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

for event in range(a_start,a_end+1):
    
    print('########## Cluster ', event, ' ##########')
    
    # # # # # # # # # # # # # # # # # # #
    # remove isolated hits
    filtered_hits = [[], [], [], [], []]   # x,y,z,t, + channel
    
    av_temp = [];
    for i in range(len(clusters[event].hitT)):
        av_temp.append(clusters[event].hitT[i])      # in order to not mess up the initial hitT[i] array    
    av_temp.sort()

    first_time = sum(av_temp[0:4])/4
    
    
    max_travel_time = 16.8   # light propagation time from one corner of the PMT rack to the other

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
    
    
    # # # # # # # # # # # # # # # # # # #
    # hit reduction - apply a more stringent cut on hit times

    # we already removed the most isolated hits from the arrays, and instead of looping over
    # every possible pair, testing to see if they could be causally connected, we can instead do
    # the following:
    
    # assume the first four points that made up the "average" initial time (above) satisfies the condition. 
    # (assume those hits can from the same, starting interaction point),
    # you can loop through each later hit and test to see if each one of those hits are causally independent
    # from the initial 4. If any are not, remove them from the array.
    
    a_t = [[], [], [], [], []];        # initial 4 hits (x,y,z,t,channel)
    
    min_filt = min(filtered_hits[3])
    
    fil_temp = []
    for i in range(len(filtered_hits[3])):
        fil_temp.append(filtered_hits[3][i])
    fil_temp.sort()
    
    for time in fil_temp:
        if len(a_t[0]) == 4:
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
            HIT[event][0].append(filtered_hits[0][i])
            HIT[event][1].append(filtered_hits[1][i])
            HIT[event][2].append(filtered_hits[2][i])
            HIT[event][3].append(filtered_hits[3][i])
            
    print('rejected ', reject_count, ' additional hits')
    
    if len(clusters[event].hitT) != (len(HIT[event][3]) + reject_count + isolated_counter):
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

# read in PDF information from fit_PDF_residual.py
PDF_params = np.loadtxt('PDF.dat', dtype = float, delimiter = None, skiprows = 1)
                
# scipy.stats exponnorm function:  exponnorm(K, loc, scale)
K_1 = PDF_params.T[0]; loc_1 = PDF_params.T[1]; scale_1 = PDF_params.T[2]


def log_prob(Vertex, X, Y, Z, T):   # first arg is the N-dim np array (position of a single "walker")
                                    # following args are constant every time the function is called (hit info)
    temp_array = []
    
    # prior --> event vertex won't be from dramatically outside tank
    if ((np.abs(Vertex[0]) < 3) and (np.abs(Vertex[1]) < 3) and (np.abs(Vertex[2]) < 3)):
    
        for ik in range(len(T)):
            # for each PMT hit in an event, find the hit time residual WRT each reconstructed vertex in the event
            tof_i = np.abs(np.sqrt( (X[ik] - Vertex[0])**2 + (Y[ik] - Vertex[1])**2 + (Z[ik] - Vertex[2])**2 ))/c

            d_t = (T[ik]/(1e9) - tof_i - Vertex[3]/c)*(1e9)               # time residual | convert to ns for the PDF

            pval = exponnorm.pdf(d_t, K=K_1, loc=loc_1, scale=scale_1)    # associated probability density function
            
            temp_array.append(np.log(pval))                               # log probability

            
        sum_temp = sum(temp_array)                                        # log likelihood
    
    else:      # return -infinity if vertex "guess" is way outside volume of detector
        
        sum_temp = -np.inf
    
    return sum_temp
        

# # # # # # # # # # # # # # # # # # # # # 
# finding vertex solution in 4-dimensions
ndim = 4                # (x,y,z,ct)

print('Number of Walkers =', nwalkers)
print(burn_in_steps, 'burn-in steps |', n_steps, 'production-run steps\n')

# final reconstructed vertex information to be saved
file1 = open('emcee_files/Mean_Acceptance_Fraction_' + str(a_start) + '_' + str(a_end) + '.dat', "w")
file2 = open('emcee_files/r_error_' + str(a_start) + '_' + str(a_end) + '.dat', "w")
file3 = open('emcee_files/reco_pos_' + str(a_start) + '_' + str(a_end) + '.dat', "w")
file4 = open('emcee_files/reco_errors_' + str(a_start) + '_' + str(a_end) + '.dat', "w")
file5 = open('emcee_files/truth_' + str(a_start) + '_' + str(a_end) + '.dat', "w")

try:
    
    for event in range(a_start,a_end+1):

        print('########## Event ' + str(event) + ' ##########')

        truth_v = [clusters[event].vtX, clusters[event].vtY, clusters[event].vtZ, 0]    # true vertex

        file2.write('count file eventNumber clusterNumber total_positional_error[cm]\n')
        file3.write('count file eventNumber clusterNumber X_reco_pos[cm] Y_reco_pos[cm] Z_reco_pos[cm] T_reco_time[ns]\n')
        file4.write('count file eventNumber clusterNumber X_error[cm] Y_error[cm] Z_error[cm] T_error[ns]\n')

        file5.write('count file eventNumber clusterNumber truth_vtX[cm] truth_vtY[cm] truth_vtZ[cm] truth_vtT[ns] energy[MeV]\n')
        file5.write(str(event) + ' ' + str(clusters[event].file_number) + ' ' + str(clusters[event].event_number) + ' ' + str(clusters[event].cluster_number) + ' ' + \
                    str(truth_v[0]*100) + ' ' + str(truth_v[1]*100) + ' ' + str(truth_v[2]*100) + ' ' + str(truth_v[3]) + ' ' + str(clusters[event].energy) + '\n')                    

        p0 = np.random.rand(nwalkers, ndim)   # initial guess
        
        for i in range(nwalkers):
            
            # (1.118573625, 3.0338736500000003)  PMT rack radius and height [m]   /   (1.5, 1.95) Tank radius and halfz height [m]

            p0[i][0] = (p0[i][0]*3) - 1.5  # water tank -radius to +radius
            p0[i][1] = (p0[i][1]*4) - 2    # water tank -height to +height
            p0[i][2] = (p0[i][2]*3) - 1.5  # water tank -radius to +radius
	
            p0[i][3] = ((p0[i][3]*2 - 1 + (clusters[event].time - 10))/(1e9))*c  # 2ns spread centered around ~10ns before the mean cluster time
            # narrow initial guess around 10 ns before the mean cluster hit time (before filtering)
            # MC says that the average hit time for low E electrons is ~10 ns after emission time t0 (from 2.5 to 60 MeV)
            
        
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
        file2.write(str(event) + ' ' + str(clusters[event].file_number) + ' ' + \
                    str(clusters[event].event_number) + ' ' + str(clusters[event].cluster_number) + ' ' + \
                    str(total_error) + '\n')
        file3.write(str(event) + ' ' + str(clusters[event].file_number) + ' ' + \
                    str(clusters[event].event_number) + ' ' + str(clusters[event].cluster_number) + ' ' + \
                    str(reco_pos[0]) + ' ' + str(reco_pos[1]) + ' ' + str(reco_pos[2]) + ' ' + str(reco_pos[3]) + '\n')
        file4.write(str(event) + ' ' + str(clusters[event].file_number) + ' ' + \
                    str(clusters[event].event_number) + ' ' + str(clusters[event].cluster_number) + ' ' + \
                    str(reco_error[0]) + ' ' + str(reco_error[1]) + ' ' + str(reco_error[2]) + ' ' + str(reco_error[3]) + '\n')
        
        print('Total Reconstructed Error: ', round(total_error,2), ' cm')

finally:            # execute even with errors (or closing the program)
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
        

print('\ndone\n')
