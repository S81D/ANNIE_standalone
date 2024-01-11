# This PDF fitter was pulled from: 
# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

# This kernal will read in the time res. data and produce a PDF with the best fit, from a list of over 100
# in the scipy library. We can use this to investigate which continous PDF will work best to model our time residual PDF.

import sys
import os
import warnings
import matplotlib
import numpy as np
import pandas as pd
from tqdm import trange
import uproot3 as uproot
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats._continuous_distns import _distn_names


# ----------------------------------------------------------- #
# first, load in MC data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
file_name = ['WCSim_Data/michel_1.ntuple.root']
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Cluster:
    def __init__(self,file_number,event_number,cluster_number,pe,time,Qb,n_hits,hitX,hitY,hitZ,hitT,hitPE,
                 vtX=None,vtY=None,vtZ=None,energy=None,t_res=None):
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
        self.vtX = vtX
        self.vtY = vtY
        self.vtZ = vtZ
        self.energy = energy
        self.t_res = t_res
        
clusters = []
        
# # # # # # # # # # # # # # #
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

    # truth information
    truth = file['phaseIITriggerTree']
    en = truth['eventNumber'].array()
    vx = truth['trueVtxX'].array()
    vy = truth['trueVtxY'].array()
    vz = truth['trueVtxZ'].array()
    ENG = truth['trueMuonEnergy'].array()
    
    
    for i in range(len(CEN)):
        if CT[i] < 100 and CN[i] == 0:
            event = Cluster(file_number=files,event_number=CEN[i],cluster_number=CN[i],pe=CPE[i],time=CT[i],
                Qb=CCB[i],n_hits=CH[i],hitX=X1[i],hitY=Y1[i],hitZ=Z1[i],hitT=T1[i],hitPE=PE1[i],
                            vtX=vx[CEN[i]]/100,vtY=vy[CEN[i]]/100,vtZ=vz[CEN[i]]/100,energy=ENG[CEN[i]]
            )
            clusters.append(event)
    

print('\nNumber of Clusters = ', len(clusters))

# ----------------------------------------------------------- #

# Calculate the hit timing residual PDF

c = 299792458  # [m/s]
c = c/(4/3)    # refractive index of water

for i in range(len(clusters)):
    TRES = []
    for j in range(len(clusters[i].hitT)):
        xx = clusters[i].hitX[j]; yy = clusters[i].hitY[j]; zz = clusters[i].hitZ[j]; tt = clusters[i].hitT[j]
        d_pos = np.abs(np.sqrt( (clusters[i].vtX - clusters[i].hitX[j])**2  +  \
                               (clusters[i].vtY - clusters[i].hitY[j])**2  +  \
                               (clusters[i].vtZ - clusters[i].hitZ[j])**2 ))
        tri = (clusters[i].hitT[j])/(1e9) - d_pos/c
        TRES.append( (tri)*(1e9) )
    
    clusters[i].t_res = TRES

# ----------------------------------------------------------- #

# Create models from data
def best_fit_distribution(data, bins=100, ax=None):
    
    """Model data by finding best fit distribution to data"""
    
    # this function is great for searching through the scipy stats library (~100 different PDFs)
    # to find the PDF which best matches the timing residual data. 
    
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        # disable conditional if statement if you want to test all PDFs
        #if ii == 71:    # non-central student's t
        if ii == 20:     # exponnorm
        
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


# transform data into panda series
temp = []
for i in range(len(clusters)):
    for j in range(len(clusters[i].t_res)):
        temp.append(clusters[i].t_res[j])
data = pd.Series(temp)


# -------------------------------------------- #

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=100, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

# Save plot limits
dataYLim = ax.get_ylim()
plt.close()

# Find best fit distribution (in this case, it is only the nct)
best_distibutions = best_fit_distribution(data, int(np.sqrt(len(data))), ax)
best_dist = best_distibutions[0]

# Make PDF with best params 
pdf = make_pdf(best_dist[0], best_dist[1])

# Display PDF
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF (' + dist_str + ')', legend=True)
data.plot(kind='hist', bins=100, density=True, alpha=0.5, label='MC Data', legend=True, ax=ax)

param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
dist_str = '{}({})'.format(best_dist[0].name, param_str)
print('\nBest-fit distribution and parameters: ',dist_str)

ax.set_title('Timing Residual PDF best fit distribution')
ax.set_xlabel('hit time residual [ns]')

path = 'hit time residual PDF (exponorm) fit _ michel e-.png'
plt.savefig(path,dpi=300, bbox_inches='tight', pad_inches=.3,facecolor = 'w')

plt.show()      # comment/uncomment if you want to display the plot
#plt.close()

file = open('PDF.dat', "w")
file.write(dist_str + '\n')    # header
for i in range(len(best_dist[1])):
    file.write(str(round(best_dist[1][i],2)) + ' ')
file.close()

print('\ndone\n')
