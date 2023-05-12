# Version 2.0 of the fit_PDF_residual.py script

# Since we are filtering hits, performing likelihood maximization w/ emcee over all of the hits may not be ideal 
# In the previous iteration, a PDF was created using all hit times
# in this version, we can construct a PDF of the filtered hit time residuals, which may lead to a better reconstruction

# also this code can be used to get the filtered hits from the emcee algorithm, without the actual reconstruction

# import packages

import sys                        # allow script to take input arguments
import numpy as np
import pandas
import warnings
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt


# path extension names for .dat files containing root tree entries
en_str = '5MeV'

# --------------------------------- #
position = 'swarm_' + en_str
folder = 'WCSim_Data/' + en_str + '/'

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
clusterTime = event_data.T[6]

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

# sort events that have an associated cluster event number
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
# Get the detector height and radius

# Read Geometry.csv file to get PMT location info
df = pandas.read_csv('FullTankPMTGeometry.csv')
x_position = []; y_position = []; z_position = []

for i in range(len(df['channel_num'])):   # loop over PMTs
    x_position.append(df['x_pos'][i]+0)
    y_position.append(df['y_pos'][i]+0.1446)
    z_position.append(df['z_pos'][i]-1.681)
    
# Find the dimensions of the detector (radius, height, etc..)
height_detector = max(y_position) - min(y_position)
radius_detector = (max(z_position) - min(z_position))/2   
        
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# MC Truth Vertex and direction information
origin = np.zeros([N_events,3])
dir_vector = np.zeros([N_events,3])
for i in range(N_events):
    origin[i][0] = vtZ[i]; dir_vector[i][0] = dirZ[i]
    origin[i][1] = vtX[i]; dir_vector[i][1] = dirX[i]
    origin[i][2] = vtY[i]; dir_vector[i][2] = dirY[i]
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

print('\nData Loaded\n')
print('#################')
print('\nN Events = ', N_events)


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


for event in range(N_events):          
    
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
    
    
# ----------------------------------------------------------- #

# Calculate the hit timing residual PDF of the filtered hits

# both vertex and HIT information is in meters (HIT[3] is in ns)

t_res_i = [[] for i in range(N_events)]
for i in range(N_events):
    for j in range(len(HIT[i][0])):
        d_pos = np.abs(np.sqrt( (vtX[i] - HIT[i][0][j])**2  +  (vtY[i] - HIT[i][1][j])**2  +  (vtZ[i] - HIT[i][2][j])**2 ))
        tri = (HIT[i][3][j])/(1e9)  -  d_pos/c   # t0 (vtT/vertex time) = 0, here -- also adjust hit times to [sec]
        t_res_i[i].append((tri)*1e9)
        
# ----------------------------------------------------------- #
import pandas as pd

# This PDF fitter was pulled from: 
# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

# Create models from data
def best_fit_distribution(data, bins=100, ax=None):
    
    """Model data by finding best fit distribution to data"""
    
    # this function is great for searching through the scipy stats library (~100 different PDFs)
    # to find the PDF which best matches the timing residual data. 
    
    # From previous investigations, it was determined that the non-central student's t distribution
    # models the WCSim low energy response response well at a range of energies (5-30 MeV).
    # Since this distribution is easily described (simple PDF equation), and I have experience using it
    # in stats class, we will use this as our sample PDF. In scipy, it is tagged as "nct".
    
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):
        
        if ii == 71:    # non-central student's t --> we'll pick out this one
        
            #print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

            distribution = getattr(st, distribution)

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                        end
                    except Exception:
                        pass

                    # identify if this distribution is better
                    best_distributions.append((distribution, params, sse))

            except Exception:
                pass

    return sorted(best_distributions, key=lambda x:x[2])


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Density Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# Load data from statsmodels datasets
t_PDF = []
for i in range(N_events):
    for j in range(len(t_res_i[i])):
        t_PDF.append(t_res_i[i][j])
data = pd.Series(t_PDF)       # transform data into panda series

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

# Save plot limits
dataYLim = ax.get_ylim()
plt.close()

# Find best fit distribution (in this case, it is only the nct)
best_distibutions = best_fit_distribution(data, 200, ax)
best_dist = best_distibutions[0]

# Make PDF with best params 
pdf = make_pdf(best_dist[0], best_dist[1])

# Display PDF
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
dist_str = '{}({})'.format(best_dist[0].name, param_str)
print('\nBest-fit distribution and parameters: ',dist_str)

ax.set_title('Timing Residual PDF best fit distribution ' + en_str + '\n' + dist_str)
ax.set_xlabel('hit time residual [s]')

path = 'hit time residual PDF (nct) fit ' + en_str + '.png'
#plt.savefig(path,dpi=300, bbox_inches='tight', pad_inches=.3,facecolor = 'w')

plt.show()      # comment/uncomment if you want to display the plot
#plt.close()

file = open('PDF.dat', "w")
file.write(en_str + ' ' + dist_str + '\n')    # header
for i in range(len(best_dist[1])):
    file.write(str(round(best_dist[1][i],2)) + ' ')
file.close()

print('\ndone\n')
