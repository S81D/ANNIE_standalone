############################################################################
# Welcome to an Event Display for WCSim Simulated Events in the ANNIE Tank #
# ------------------------------------------------------------------------ #
#    The Event Display shows the tank PMT's response to simulated events   #
#  events from WCSim. The Timing and Hit information extracted from WCSim  #
#                 are displayed in 2D Event Display Plots.                 #
#    MC Truth information is also displayed about the primary particle.    #    
# ------------------------------------------------------------------------ #
#                         Author: Steven Doran                             #
#                   Last Date of Modification: August 2023                 #
############################################################################

import sys                      
import numpy as np
import uproot3 as uproot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas


# ------------------------------------------------------------------- #
file = uproot.open('MC_Data/muon_swarm/muon_swarm.ntuple.root')
# ------------------------------------------------------------------- #

primary_particle = r'$\mu^-$'              # MC particle type

How_many_clusters = 1                      # How many events/clusters should be displayed/saved?

cluster_offset = 0                         # Adjust to plot a later cluster 
                                           # (this will be the first event/cluster)

Name = '2D_Plots/MC muon ANNIEEvent '      # Name of the Plot/Event?

PMT_marker_size = 200                      # default is 200

# ------------------------------------------------------------------- #


print('\nLoading Data...\n')


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

# record how many clusters per WCSim event
clusters_per_event = [(clustereventNumber.tolist()).count(i) for i in eventNumber]
cluster_count = [clusters_per_event[int(en)] for en in clustereventNumber]

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# MC Truth Vertex and direction information
origin = np.zeros([N_clusters,3])
dir_vector = np.zeros([N_clusters,3])
for i in range(N_clusters):
    origin[i][0] = vtZ[i]; dir_vector[i][0] = dirZ[i]
    origin[i][1] = vtX[i]; dir_vector[i][1] = dirX[i]
    origin[i][2] = vtY[i]; dir_vector[i][2] = dirY[i]

#######################################################################
# We can also create some custom arrays

# Charge and hits
Sum_PE = [[] for i in range(N_clusters)]             # summed P.E. on each PMT
hits_per_PMT = [[] for i in range(N_clusters)]       # number of hits on each PMT
unique_PMTs = [[] for i in range(N_clusters)]        # unique hit PMTs
for i in range(N_clusters):
    u, c = np.unique(Channel[i], return_counts=True)
    unique_PMTs[i].append(u.tolist())                # prevents this list from becoming a numpy array
    hits_per_PMT[i].append(c.tolist())               # (only included because we call the method index() later on, which only works on lists)
for i in range(N_clusters):
    for j in range(len(unique_PMTs[i][0])):
        pe = 0.
        for k in range(len(Channel[i])):
            if unique_PMTs[i][0][j] == Channel[i][k]:
                pe = pe + hitPE[i][k]
        Sum_PE[i].append(pe)
        
# record how many clusters per WCSim event
clusters_per_event = [(clustereventNumber.tolist()).count(i) for i in eventNumber]
cluster_count = [clusters_per_event[int(en)] for en in clustereventNumber]
        
# We can define whether an event would ideally trigger an extended readout:
ext_readout_trig = []
for i in range(N_clusters):
    if clusterMaxPE[i] > 7.:
        ext_readout_trig.append(True)
    else:
        ext_readout_trig.append(False)


###################################################
### Load in and Construct the Detector Geometry ###

# Read Geometry.csv file to get PMT location info
df = pandas.read_csv('FullTankPMTGeometry.csv')

channel_number = []; location = []; panel_number = []
x_position = []; y_position = []; z_position = []

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
########### Construct the 2D Event Display ############
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

print('\nSaving Plots...\n')

# We must first unfold the 2D geometry of the tank

# # # # # # # # # # # # # # # # # # # #
# Split Barrel PMTs from Top/Bottom
barrel_index = []; Top_index = []; Bottom_index = []
for i in range(len(location)):
    if location[i] == 'Barrel':
        barrel_index.append(i)
    elif location[i] == 'Bottom':
        Bottom_index.append(i)
    elif location[i] == 'Top':
        Top_index.append(i)
        
# For the barrel PMTs, we can find the distance from the PMT to the 'Top' PMTs (in y) - this will serve as one
# of the dimensions. We can get the other dimension through an angle phi relative to the centerpoint and some
# fixed point along the barrel. From there we know the radius and can calculate an arclength (other dimension).

# Take point (z = 0, x = min(x_position)) as the fixed reference point (phi = 0), as it is between two panels.
# The angle will be in the Z-X plane. Use law of cosines to find the angle phi --> cos(phi) = (r^2 + a^2 + x^2)/(2ra)
# where:   
# r = distance between ref point and center, a = distance between PMT and center,
# x = distance between PMT and ref point

# Since the angle will run from [0,pi], we will have degeneracy - we can use the panel numbers to distinguish each side of the barrel.
# --> Based on the geometry of the tank, define PMTs on panels 3, 4, 5, 6 as being one set of angles ("right" side)
#     PMTs 2, 1, 8, 7 will be the other. We can then artifically add (-) to one set of PMT angles to get a full range [0,2pi]

ref_point = [0, min(x_position)]    # [z,x]
ref_radius = np.sqrt((ref_point[0]-0)**2 + (ref_point[1]-0)**2)

# find the "radius" of the Tank (avg distance from PMTs to centerpoint, in Z-X plane) (centerpoint is 0,0,0)
sum_radius = 0.
for i in range(len(channel_number)):
    if i in barrel_index:
        sum_radius = sum_radius + np.sqrt((x_position[i]**2) + (z_position[i]**2))
        
radius = sum_radius/len(barrel_index)              # this will be used to calculate the arclength once we have phi
                                                   # (assume all PMTs along the barrel share this avg radius (a good approx.))

left_panels = [1,2,7,8]                            # categorize which panels are "left" vs "right"

barrel_angles = [];
for i in range(len(channel_number)):
    if i in barrel_index:
        x = np.sqrt(((z_position[i] - ref_point[0])**2) + ((x_position[i] - ref_point[1])**2))
        a = np.sqrt((x_position[i]**2) + (z_position[i]**2))
        r = ref_radius
        cos_phi = ((r**2)+(a**2)-(x**2))/(2*r*a)   # law of cosines, solving for cos(phi)
        phi = np.arccos(cos_phi)                   # units of radians
            
        if panel_number[i] in left_panels:         # Break barrel into "left" and "right" PMTs
            barrel_angles.append(-phi)             
        else:
            barrel_angles.append(+phi)             
    else:
        barrel_angles.append('Not Barrel')         

# convert into a linear distance (arclength = r*phi) away from the ref PMT, for barrel PMTs
dist_barrel = []
for i in range(len(channel_number)):
    if i in barrel_index:
        s = barrel_angles[i]*radius
        dist_barrel.append(s)
    else:
        dist_barrel.append('Not Barrel')

# Do the other dimension (y)
barrel_height = []
for i in range(len(channel_number)):
    if i in barrel_index:
        barrel_height.append(y_position[i])
    else:
        barrel_height.append('Not Barrel')

        
# Now we can do the Top and Bottom PMTs

# To make the event display "looking" from the inside, you need to flip the x and z coords.
# Currently, the top and bottom are not configured to match up with the barrel

Top_z = []; Top_x = []; Bottom_z = []; Bottom_x = []
for i in range(len(channel_number)):
    if i in Top_index:
        Top_z.append(-z_position[i]); Top_x.append(x_position[i])     # CHANGED
        Bottom_z.append('Not Bottom'); Bottom_x.append('Not Bottom')
    elif i in Bottom_index:
        Top_z.append('Not Top'); Top_x.append('Not Top')
        Bottom_z.append(-z_position[i]); Bottom_x.append(-x_position[i])  # CHANGED
    else:
        Top_z.append('Not Top'); Top_x.append('Not Top')
        Bottom_z.append('Not Bottom'); Bottom_x.append('Not Bottom')


# The event display will show you an unfolded 2D view of the tank as if you were
# standing in the center, looking towards one side of the tank.

#####################################################################

# # # # # # # # # # # # #
# Displaying Tank Event #
# # # # # # # # # # # # #

print('\n###########################')

# # # # # # # # # # # # # # # # # # # # # # # 
# Build the Figure

def BuildEventDisplay2D(N_Hits,Total_Charge,En,Charge,Time,origin_x,origin_y,origin_z,DV_x,DV_y,DV_z,Qb,maxpe,exttrig):

    ##########################################################
    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.tick_params(labelbottom=False, labelleft=False)

    fig = plt.figure(figsize = (10,10))#, constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_barrel = fig.add_subplot(gs[1, :])
    ax_bottom = fig.add_subplot(gs[2, 1])
    
    # subplot figure for text and title
    ax_text = fig.add_subplot(gs[0,0])
    ax_text.axis('off')
    # Add Statistics on PE's
    hits_text = str(N_Hits) + ' hits / ' + str(round(Total_Charge,2)) + ' p.e.'
    ax_text.text(0.15,0.1,hits_text,size = 12,transform = ax_text.transAxes)

    # Add a Description and Title
    title = 'ANNIE Simulated Event'
    ax_text.text(0.15, 0.75,title,size=15,color='darkred',transform=ax_text.transAxes)
    description = 'Event Description:'
    ax_text.text(0.15, 0.6,description,size=12,color='darkred',transform=ax_text.transAxes)
    ax_text.text(0.15, 0.5,'Primary ',size=12,color='darkred',transform=ax_text.transAxes)
    ax_text.text(0.45, 0.5,primary_particle,size=12,color='darkred',transform=ax_text.transAxes,fontweight = 'bold')
    e_description = 'E = ' + str(round(En,2)) + ' MeV'
    ax_text.text(0.15, 0.4,e_description,size=12,color='darkred',transform=ax_text.transAxes)
    
    c_description = 'Cluster ' + str(int(clusterNumber[i]+1)) + '/' + str(cluster_count[i])
    ax_text.text(0.15, 0.2,c_description,size=12,transform=ax_text.transAxes)

    #####################################################################################
    # Adding histograms displaying the charge and timing distributions of the given event
    ax_hist = fig.add_subplot(gs[0,2])
    ax_hist.axis('off')
    # # # # # # # # # #
    ax_charge = ax_hist.inset_axes([0.15, 0.6, 0.7, 0.3])
    ax_charge.set_title('P.E. per Hit')
    ax_charge.hist(Charge, bins = 'auto')   # allows matplotlib to chose between available options --> see documentation
    # # # # # # # # # #
    ax_time = ax_hist.inset_axes([0.15, 0.1, 0.7, 0.3])
    ax_time.set_title('Hit Timing [ns]')
    ax_time.hist(Time, bins = 'auto')

    ######################################################################################
    # add model of tank to indicate vertex position and direction of primary particle
    
    # rectangle: looking at it w/ x-axis = z, [- to +], y-axis = y, [- to +]
    # circle: looking from the top, down w/ x-axis = z, y-axis = x (both - to +)
    
    ax_tank = fig.add_subplot(gs[2,2])
    ax_tank.axis('off')
    circle_tank = plt.Circle((0.77,0.45), (.15))
    ax_tank.add_patch(circle_tank)
    rect_tank = plt.Rectangle((0.25,0.20), 0.35, 0.5)   # (x,y coords gives the corner of the rectangle)
    ax_tank.add_patch(rect_tank)
    
    # Position of the vertex
    scale = 0.15/radius
    # origin (z,x,y)
    z_psuedo = 0.77 + origin_z*scale; x_psuedo = 0.45 + origin_x*scale
    ax_tank.scatter(z_psuedo, x_psuedo, marker = '*', color = 'black')
    
    scale_z = 0.35/(radius*2); scale_y = 0.5/(3)  # height scaling
    z_psuedo_r = 0.25 + (0.35/2) + origin_z*scale_z; y_psuedo_r = 0.20 + (0.5/2) + origin_y*scale_y
    ax_tank.scatter(z_psuedo_r, y_psuedo_r, marker = '*', color = 'black')
    
    ver = 'Truth Vertex & Direction'
    ax_tank.text(0.20, 0.75,ver,size=12,color='black',transform=ax_tank.transAxes)
    
    ax_tank.set_xlim([0,1])
    ax_tank.set_ylim([0,1])
    
    # Direction of primary particle (direction_vector[x,y,z])
    ax_tank.quiver(z_psuedo, x_psuedo, DV_z, DV_x, color = 'black')#, scale_units='xy', scale = 12)  # arbitrary scaling
    ax_tank.quiver(z_psuedo_r, y_psuedo_r, DV_z, DV_y, color = 'black')#, scale_units='xy', scale = 2.5)
    ######################################################################################
    # Charge Balance and Max PE
    ax_qb = fig.add_subplot(gs[2,0])
    ax_qb.axis('off')
    
    chargebalance = r'$Q_{b}$' + ' = ' + str(round(Qb,3))
    ax_qb.text(0.15, 0.8,chargebalance,size=15,color='black',transform=ax_qb.transAxes)
    
    maximumpe = r'$PE_{max}$' + ' = ' + str(round(maxpe,2))
    ax_qb.text(0.15, 0.65,maximumpe,size=15,color='black',transform=ax_qb.transAxes)

    if exttrig == True:
        ext_text = 'Extended Readout\nTriggered (> 7 p.e.)'
    elif exttrig == False:
        ext_text = 'No Extended Readout\nTriggered (< 7 p.e.)'
    ax_qb.text(0.15, 0.4,ext_text,size=12,color='black',transform=ax_qb.transAxes)
    
    ######################################################################################
    format_axes(fig)   # remove tick markers

    # Draw top and bottom black circles --> these will serve as our subplots
    circle_top = plt.Circle((0,0), (1), color='black', zorder = 1)
    ax_top.add_patch(circle_top)

    circle_bottom = plt.Circle((0,0), (1), color='black', zorder = 1)
    ax_bottom.add_patch(circle_bottom)

    # Turn off Top and Bottom (and text) axes --> will replace top/bottom with black circles
    ax_bottom.axis('off')
    ax_top.axis('off')
    
    # Set the PMT barrel plot backgrounds black
    ax_barrel.set_facecolor('black')
    
    return ax_top, ax_barrel, ax_bottom, ax_text, fig

#############################

for i in range(cluster_offset, How_many_clusters + cluster_offset):
    
    print('\nEvent ' + str(int(clustereventNumber[i])) + \
          ' | Cluster ' + str(int(clusterNumber[i]+1)) + '/' + str(cluster_count[i]))
    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ############################ Charge Plot ##############################
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    ax_top, ax_barrel, ax_bottom, ax_text, fig = BuildEventDisplay2D(clusterHits[i],clusterPE[i],Energy[i],hitPE[i],hitT[i],
                                                               origin[i][1],origin[i][2],origin[i][0],dir_vector[i][1],
                                                                dir_vector[i][2],dir_vector[i][0],clusterChargeBalance[i],
                                                               clusterMaxPE[i],ext_readout_trig[i])
    ax_text.text(0.15, -0.1,'Charge Plot',size=15,color='black',transform=ax_text.transAxes,fontweight = 'bold')

    # If the top or bottom are not hit, the plots do not scale correctly - add one black scatterpoint to scale correctly
    ax_top.scatter(Top_z[0], Top_x[0], s = PMT_marker_size, color = 'black')
    ax_bottom.scatter(Bottom_z[0], Bottom_x[0], s = PMT_marker_size, color = 'black')
    
    # For the Charge Plot, we will display the summed charge per PMT, since we may have multiple hits per PMT
    # Ensure the colorbar is scaled by the brightest PMT (take vmax = max(pmt_charge))
    for hit in range(len(Channel[i])):  # loop through hit PMTs
        index = unique_PMTs[i][0].index(Channel[i][hit])
        index_id = channel_number.index(Channel[i][hit])
        if Top_z[index_id] != 'Not Top':
            ax_top.scatter(Top_z[index_id], Top_x[index_id], s = PMT_marker_size, c = Sum_PE[i][index], cmap=plt.cm.plasma,
                           vmin = 0, vmax = max(Sum_PE[i]), alpha = 1)
            if hits_per_PMT[i][0][index] > 1:
                ax_top.annotate(hits_per_PMT[i][0][index], (Top_z[index_id]+0.06, Top_x[index_id]+0.06), color = 'white')
        if Bottom_z[index_id] != 'Not Bottom':
            ax_bottom.scatter(Bottom_z[index_id], Bottom_x[index_id], s = PMT_marker_size, c = Sum_PE[i][index], cmap=plt.cm.plasma,
                              vmin = 0, vmax = max(Sum_PE[i]), alpha = 1)
            if hits_per_PMT[i][0][index] > 1:
                ax_bottom.annotate(hits_per_PMT[i][0][index], (Bottom_z[index_id]+0.06, Bottom_x[index_id]+0.06), color = 'white')
        if dist_barrel[index_id] != 'Not Barrel':
            p = ax_barrel.scatter(dist_barrel[index_id],barrel_height[index_id], s = PMT_marker_size, c = Sum_PE[i][index],
                                  cmap=plt.cm.plasma, vmin = 0, vmax = max(Sum_PE[i]), alpha = 1)
            if hits_per_PMT[i][0][index] > 1:
                ax_barrel.annotate(hits_per_PMT[i][0][index], (dist_barrel[index_id]+0.07, barrel_height[index_id]+0.07), color = 'white')

    # Set barrel plot y-limits to the height of the top and bottom PMT rack
    ax_barrel.set_ylim([min(y_position),max(y_position)])

    # set colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])    # (horizontal position, vert position, bar thickness, bar length)
    fig.colorbar(p, ax=ax_top, label = 'Charge [p.e.]', cax = cbar_ax)

    path = Name + str(int(clustereventNumber[i])) + ' Cluster ' + str(int(clusterNumber[i]+1)) + ' Charge Plot.png'
    plt.savefig(path,dpi=300, bbox_inches='tight', pad_inches=.3,facecolor = 'w')
    
    plt.close()


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ############################ Timing Plot ##############################
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    ax_top, ax_barrel, ax_bottom, ax_text, fig = BuildEventDisplay2D(clusterHits[i],clusterPE[i],Energy[i],hitPE[i],hitT[i],
                                                               origin[i][1],origin[i][2],origin[i][0],dir_vector[i][1],
                                                                dir_vector[i][2],dir_vector[i][0],clusterChargeBalance[i],
                                                               clusterMaxPE[i], ext_readout_trig[i])
    
    ax_text.text(0.15, -0.1,'Timing Plot',size=15,color='black',transform=ax_text.transAxes,fontweight = 'bold')

    # If the top or bottom are not hit, the plots do not scale correctly - add one black scatterpoint to scale correctly
    ax_top.scatter(Top_z[0], Top_x[0], s = PMT_marker_size, color = 'black')
    ax_bottom.scatter(Bottom_z[0], Bottom_x[0], s = PMT_marker_size, color = 'black')
    

    # Timing Plot
    for hit in range(len(Channel[i])):  # loop through hit PMTs
        index = unique_PMTs[i][0].index(Channel[i][hit])
        index_id = channel_number.index(Channel[i][hit])
        if Top_z[index_id] != 'Not Top':
            ax_top.scatter(Top_z[index_id], Top_x[index_id], s = PMT_marker_size, c = hitT[i][hit], cmap=plt.cm.cool,
                           vmin = 0, vmax = max(hitT[i]), alpha = 1)
            if hits_per_PMT[i][0][index] > 1:
                ax_top.annotate(hits_per_PMT[i][0][index], (Top_z[index_id]+0.06, Top_x[index_id]+0.06), color = 'white')
        if Bottom_z[index_id] != 'Not Bottom':
            ax_bottom.scatter(Bottom_z[index_id], Bottom_x[index_id], s = PMT_marker_size, c = hitT[i][hit], cmap=plt.cm.cool,
                              vmin = 0, vmax = max(hitT[i]), alpha = 1)
            if hits_per_PMT[i][0][index] > 1:
                ax_bottom.annotate(hits_per_PMT[i][0][index], (Bottom_z[index_id]+0.06, Bottom_x[index_id]+0.06), color = 'white')
        if dist_barrel[index_id] != 'Not Barrel':
            p = ax_barrel.scatter(dist_barrel[index_id],barrel_height[index_id], s = PMT_marker_size, c = hitT[i][hit],
                                  cmap=plt.cm.cool, vmin = 0, vmax = max(hitT[i]), alpha = 1)
            if hits_per_PMT[i][0][index] > 1:
                ax_barrel.annotate(hits_per_PMT[i][0][index], (dist_barrel[index_id]+0.07, barrel_height[index_id]+0.07), color = 'white')

    # Set barrel plot y-limits to the height of the top and bottom PMT rack
    ax_barrel.set_ylim([min(y_position),max(y_position)])

    # set colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])    # (horizontal position, vert position, bar thickness, bar length)
    fig.colorbar(p, ax=ax_top, label = 'Time [ns]', cax = cbar_ax)

    path = Name + str(int(clustereventNumber[i])) + ' Cluster ' + str(int(clusterNumber[i]+1)) + ' Timing Plot.png'
    plt.savefig(path,dpi=300, bbox_inches='tight', pad_inches=.3,facecolor = 'w')
    
    plt.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#######################################################################

print('\ndone\n')  
