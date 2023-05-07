# This PDF fitter was pulled from: 
# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

# This kernal will read in the time res. data and produce a PDF with the best fit, from a list of over 100
# in the scipy library. We can use this to investigate which continous PDF will work best to model our time residual PDF.

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt


# ----------------------------------------------------------- #
# first, load in MC data

# # # # # # # # # # # # # # # 
# Parameterization -- load in the .dat files exported from TA root files
en_str = '30MeV'                 # specifies the title, output file header, and path for input files
position = 'swarm_' + en_str
folder = 'WCSim Data/' + en_str + '/'

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
        
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# MC Truth Vertex and direction information
origin = np.zeros([N_events,3])
dir_vector = np.zeros([N_events,3])
for i in range(N_events):
    origin[i][0] = vtZ[i]; dir_vector[i][0] = dirZ[i]
    origin[i][1] = vtX[i]; dir_vector[i][1] = dirX[i]
    origin[i][2] = vtY[i]; dir_vector[i][2] = dirY[i]

# ----------------------------------------------------------- #

# Calculate the hit timing residual PDF

c = 299792458  # [m/s]i
c = c/(4/3)    # refractive index of water

t_res_i = [[] for i in range(N_events)]
for i in range(N_events):
    for j in range(len(hitT[i])):
        d_pos = np.abs(np.sqrt( (vtX[i] - hitX[i][j])**2  +  (vtY[i] - hitY[i][j])**2  +  (vtZ[i] - hitZ[i][j])**2 ))
        tri = (hitT[i][j])/(1e9)  -  d_pos/c   # t0 (vtT/vertex time) = 0, here -- also adjust hit times to [sec]
        t_res_i[i].append((tri)*1e9)

# ----------------------------------------------------------- #

# Create models from data
def best_fit_distribution(data, bins=100, ax=None):
    
    """Model data by finding best fit distribution to data"""
    
    # this function is great for searching through the scipy stats library (~100 different PDFs)
    # to find the PDF which best matches the timing residual data. 
    
    # From previous investigations, it was determined that the non-central student's t distribution
    # models the WCSim low energy response response well at a range of energies (5-15 MeV, waiting on 30).
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
print('Best-fit distribution and parameters: ',dist_str)

ax.set_title('Timing Residual PDF best fit distribution ' + en_str + '\n' + dist_str)
ax.set_xlabel('hit time residual [s]')

path = 'hit time residual PDF (nct) fit ' + en_str + '.png'
#plt.savefig(path,dpi=300, bbox_inches='tight', pad_inches=.3,facecolor = 'w')

#plt.show()      # comment/uncomment if you want to display the plot
plt.close()

file = open('PDF.dat', "w")
file.write(en_str + dist_str + '\n')    # header
for i in range(len(best_dist[1])):
    file.write(str(round(best_dist[1][i],2)) + ' ')
file.close()

print('\ndone')
