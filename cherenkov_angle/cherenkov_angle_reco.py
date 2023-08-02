# Quick, dirty cherenkov angle reconstruction for WCSim generated MC events

# This script can generate a distribution of reconstructed cherenkov angles across events.
# It will also export the reconstructed angles for each event/cluster to a .txt file.

# Author: Steven Doran     August 2023

import sys                      
import numpy as np
import uproot3 as uproot
import matplotlib.pyplot as plt
import pandas
from tqdm import trange
import itertools
import math
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


# ------------------------------------------------------------------- #
file = uproot.open('WCSim_Data/10MeV/electron_swarm_10MeV.ntuple.root')
# ------------------------------------------------------------------- #

save_info = True
txtfilename = 'electron_reco_angles.dat'


create_distribution = True
dist_filename = 'electron swarm 10MeV reco cherenkov angle dsitribution.png'
particle = r'$\mu^-$'
dist_color = 'moccasin'                      # ['moccasin', 'darksalmon', 'lightsteelblue', 'lightsalmon']


print('\nLoading data...')


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

print('\n#################')
print('\nNumber of Events = ', N_events)
print('Number of Clusters = ', N_clusters)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# specified functions


# Using 3 PMT hits, define a plane and circle containing and encompassing the 3 points (hits)

def C_circle(A, B, C):  # arguments A, B, and C = hit (X,Y,Z) coordinates (3D arrays)
    
    # Perform axis transformation
    u1 = []; C_minus_A = []
    for i in range(3):
        u1.append( B[i] - A[i] )
        C_minus_A.append( C[i] - A[i] )
        
    w1 = np.cross(C_minus_A, u1)
    u = u1/np.sqrt( (u1[0])**2 + (u1[1])**2 + (u1[2])**2 )    # new x-axis unit vector
    w = w1/np.sqrt( (w1[0])**2 + (w1[1])**2 + (w1[2])**2 )    # new z-axis unit vector (normal vector)
    v = np.cross(w, u)                                        # new y-axis unit vector
    
    # We now have three orthogonal unit vectors, with u and v spanning the plane
    
    # Find the new 2D coordinates of the 3 hits.
    # We do this by finding the dot products of (B-A) and (C-A) with u and v
    # b = (b_x, 0); c = (cx, cy)
    b = [np.dot(u1, u), 0]
    c = [np.dot(C_minus_A, u), np.dot(C_minus_A, v)]
    
    # The center of the circle encompased by the 3 hits (points) must lie on the line x = bx/2
    # --> we will label this point (bx/2, h)
    # The distance from c must be the same as the distance from the origin
    # (cx - bx/2)**2 + (cy - h)**2 = (bx/h)**2 + h**2
    # --> h = (cx - bx/2)**2 + cy**2 - (bx/2)**2 / (2*cy)
    h = ( (c[0] - b[0]/2)**2 + c[1]**2 - (b[0]/2)**2 ) / (2*c[1])
    center2D = [b[0]/2, h]
    
    # Find the position of the center (transformed back into 3D)
    center = []
    for i in range(3):
        center.append( A[i] + (b[0]/2)*u[i] + h*v[i] )
        
    # Verify that all points (hits) are equidistant from the center of the circle
    p1_dist = np.sqrt( (center[0] - A[0])**2 + (center[1] - A[1])**2 + (center[2] - A[2])**2 )
    p2_dist = np.sqrt( (center[0] - B[0])**2 + (center[1] - B[1])**2 + (center[2] - B[2])**2 )
    p3_dist = np.sqrt( (center[0] - C[0])**2 + (center[1] - C[1])**2 + (center[2] - C[2])**2 )
    
    # Height of vertex above plane; need to first solve for eqn of the plane
    k = -w[0]*A[0] - w[1]*A[1] - w[2]*A[2]    # in principle you can use any PMT hit (any point)
    h = np.abs(w[0]*vertex[0] + w[1]*vertex[1] + w[2]*vertex[2] + k)/(np.sqrt(w[0]**2 + w[1]**2 + w[2]**2))
    
    terminate = False
    
    if round(p1_dist,1) == round(p2_dist,1) == round(p3_dist,1):
        radius = p1_dist
    
    # These values are not equal when there is the same PMT hit or the points lie along the same line
    else:
        terminate = True
        radius = 0
    
    return center, radius, h, terminate



# Using the vertex (truth or reconstructed) and the center + radius of the cherenkov circle, calculate the angle

def C_angle(origin,center,radius,height):
    
    # find the distance from center of cherenkov circle to vertex
    l = np.sqrt( (center[0] - origin[0])**2 + (center[1] - origin[1])**2 + (center[2] - origin[2])**2 )
    
    # angle is as simple as tan(theta) = radius/l
    angle = np.rad2deg(np.arctan(radius/l))
    
    
    return angle



# Hit Filtering functions


# based on time residuals

def time_selection(TW,origin,hitX,hitY,hitZ,hitT):
    
    c = 299792458  # [m/s]
    c = c/(4/3)    # refractive index of water
    
    # calculate hit timing residuals
    t_res_i = []
    for i in range(len(hitX)):
        d_pos = np.abs(np.sqrt( (origin[0] - hitZ[i])**2  + \
                                   (origin[1] - hitX[i])**2  +  (origin[2] - hitY[i])**2 ))
        tri = (hitT[i])/(1e9)  -  d_pos/c   # t0 = 0 for MC
        t_res_i.append(tri*1e9)
    
    # sort the timing residuals, keep the index ordering
    sorted_index = np.argsort(t_res_i)   # contains the true indicies of the hitT array, sorted by residuals
    t_res_i.sort()
    
    time_window = TW     # hit time allowance
    
    t_start = math.floor(min(t_res_i)); t_end = math.ceil(max(t_res_i))
    count = [[], [], []]    
    for i in range(t_start, t_end, 1):    # 1 ns windows
        lw = i; hw = i+TW
        count[0].append(lw); count[1].append(hw)
        
        #lw = t_res_i[i]; hw = t_res_i[i] + time_window
        #count[0].append(lw); count[1].append(hw)
        temp = []
        for j in range(len(t_res_i)):     # not an error, looping over the array twice
            if hw >= t_res_i[j] >= lw:
                temp.append(t_res_i[j])
        count[2].append(len(temp))
        
        if hw > t_end:
                break
            
    in_val = count[2].index(max(count[2]))     # preference windows with lower time residuals (select first appearence of max value)
    
    save = []
    for i in range(len(t_res_i)):
        if count[0][in_val] <= t_res_i[i] <= count[1][in_val]:
            save.append(sorted_index[i])
    
    return save


# Grab the top 10 charged hits (what we use)

def charge_selection(N_hits,hitPE):
    
    new_array = [hitPE[i] for i in range(len(hitPE))]
    
    sorted_index = np.argsort(new_array)
    save = sorted_index[-N_hits:]
    
    return save


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #



HIT = [[] for i in range(N_clusters)]        # final, filtered, reduced hits list
for i in range(len(HIT)):
    for j in range(5):
        HIT[i].append([])
        
    
print('\nFiltering hits...')
for event in range(N_clusters):
    
    filtered_hits = [[], [], [], [], [], [], []]   # x,y,z,t, + channel + p.e. + origin
    
    hit_indices = charge_selection(10, hitPE[event])
    
    
    for i in range(len(hitT[event])):
        
        if (i in hit_indices):
            
            if Channel[event][i] not in filtered_hits[4]:    # discard multiple hits on the same PMT

                filtered_hits[3].append(hitT[event][i])

                filtered_hits[0].append(hitX[event][i])      # to keep the indexing consistent
                filtered_hits[1].append(hitY[event][i])
                filtered_hits[2].append(hitZ[event][i])

                filtered_hits[4].append(Channel[event][i])   

                filtered_hits[5].append(hitPE[event][i])



                filtered_hits[6].append(origin[event])
                
    
    for i in range(len(filtered_hits[0])):
        
        HIT[event][0].append(filtered_hits[0][i])
        HIT[event][1].append(filtered_hits[1][i])
        HIT[event][2].append(filtered_hits[2][i])
        HIT[event][3].append(filtered_hits[3][i])
        HIT[event][4].append(filtered_hits[4][i])




# Generate 3-hit PMT combinations for each event
# Note that all hits must be unique (no repeated PMT hits)

saved_combos = [[] for i in range(N_clusters)]

print('\nGenerating 3-hit PMT combinations...')

for i in trange(N_clusters):
    
    if clusterTime[i] < 20 and clusterHits[i] > 10:
    
        a = [ix for ix in range(len(HIT[i][0]))]    # n possible hits (assigned by index)
        b = list(itertools.combinations(a,3))       # all possible 3-hit combinations

        for j in range(len(b)):   # loop over 3-hit combinations

            break_it = False

            # check if PMT hits are unique
            pmt_ids = 0
            for k in range(3):
                check = HIT[i][4][b[j][k]]
                if pmt_ids == check:
                    break_it = True
                else:
                    pmt_ids = HIT[i][4][b[j][k]]

            if break_it == True:
                continue   # do not append the 3-hit PMT combo
            else:

                saved_combos[i].append(b[j])



# Reconstruct cherenkov angle

angle_dump = [[] for i in range(N_clusters)]

print('\nReconstructing Cherenkov Angles...')

for e in trange(N_clusters):
    
    if len(saved_combos[e]) != 0:
    
        for i in range(len(saved_combos[e])):

            hits = saved_combos[e][i]

            p1 = [HIT[e][2][hits[0]], HIT[e][0][hits[0]], HIT[e][1][hits[0]]]
            p2 = [HIT[e][2][hits[1]], HIT[e][0][hits[1]], HIT[e][1][hits[1]]]
            p3 = [HIT[e][2][hits[2]], HIT[e][0][hits[2]], HIT[e][1][hits[2]]]
            
            vertex = origin[e]

            center,radius,height,terminate = C_circle(p1,p2,p3)

            if terminate == False:

                angle = C_angle(vertex, center, radius, height)
                angle_dump[e].append(angle)




# Select the most common cherenkov angle from each distribution

binning = np.arange(0, 91, 3)

dummy = []; other = []
for i in range(N_clusters):
    
    if len(angle_dump[i]) != 0:

        # counts value will represent the lower bound for the bin ("0" = 0-1, "5" = 5-6, "31" = 31-32)
        counts, bins = np.histogram(angle_dump[i], bins = binning)
        
        dum = [[], []]
        for j in range(1, len(counts) - 1):
            sum_e = counts[j-1] + counts[j] + counts[j+1]    # sum each neighboring bin
            dum[0].append(bins[j])
            dum[1].append(sum_e)
        
        # find the angle with the maximum number of counts
        dum_max = max(dum[1])
        temp_dum = []
        for j in range(len(dum[1])):
            if dum[1][j] == dum_max:
                # assign cherenkov angle of the event
                temp_dum.append(dum[0][j])
        dummy.append(np.average(temp_dum))
        other.append(i)   # record index


if save_info == True:
    txtfile = open(txtfilename, "w")

    txtfile.write('eventNumber clusterNumber angle\n')     # header
    for i in range(len(other)):
        txtfile.write(str(clustereventNumber[other[i]]) + ' ' + str(clusterNumber[other[i]]) + ' ' + str(dummy[i]) + '\n')
    txtfile.close()
        

if create_distribution == True:
    
    print('\nShowing Reconstructed Cherenkov Angle Distribution...')

    n_entries = len(dummy)
    performance = 0
    for i in range(n_entries):
        if dummy[i] < 30:
            performance += 1
    performance = performance/n_entries

    counts, bins = np.histogram(dummy, bins = binning)
    bin_edges = binning[:-1]

    fig, ax = plt.subplots()

    plt.hist(bin_edges, binning, weights = counts, color = dist_color)
    

    # Label 42-degrees and/or 27-degrees for muon/electrons

    plt.axvline(42, color = 'black', linestyle = 'dashed')
    ax.text(.5,.9,r'$\theta_c = 42^\circ$',size = 12,transform = ax.transAxes)

    #plt.axvline(27, color = 'blue', linestyle = 'dashed')
    #ax.text(.06,.9,r'$\theta_{\mu,max} = 27^\circ$',size = 12,color = 'blue',transform = ax.transAxes)


    ax.text(.6,.75,'Events with ' + r'$\theta_c < 30^{\circ} = $' + str(round(performance*100,2)) + '%',size = 8,transform = ax.transAxes)
    ax.text(.75,.65,'entries = ' + str(n_entries),size = 8,transform = ax.transAxes)
    
    plt.title(r'$\theta_{c,reco}$' + ' for WCSim ' + particle +
            ' events (center, E = ' + str(round(Energy[0],2)) + ' MeV)')       
    
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    plt.xlabel('Reconstructed Cherenkov Angle (' + r'$\theta_c$' + ') [degrees]')
    plt.savefig(dist_filename,dpi=300,bbox_inches='tight', pad_inches=.3,facecolor = 'w')
    plt.show()

    print('\nPercentage of events with angle < 30 degrees: ' + str(round(performance*100,2)) + '%\n')


print('\ndone\n')



