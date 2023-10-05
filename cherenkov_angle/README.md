# Cherenkov Angle Calculation

A quick and dirty cherenkov angle reconstruction for MC events in the ANNIE Tank.

### Details

By reconstructing the cherenkov angle of an event within the ANNIE water volume, it is possible to identify different classes of low energy particles.

The method used for this code closely follows from the methods used by T2K in performing their NCQE-XS measurement. 

The size of the tank makes reconstructing cherenkov cones difficult in ANNIE, as the granularity and coverage of the PMTs is much less than that of Super-K.
Vertices are also much more likely to be near (< 1 meter) a PMT, leading to smearing and elongation of the cone. 
In the water tank, cherenkov cones appear more like a disk, illuminating a "blob" of PMTs, rather than outlining a distinct ring-like profile. Therefore,
a straight forward calculation of the cherenkov angle is difficult. To reconstruct cherenkov cones within ANNIE, this code take 3-hit combinations
of the PMTs to reconstruct a circle containing those 3 points. An angle is then calculated WRT the radius of the cherenkov circle by using the truth (and eventually the reconstructed)
vertex. Doing this over all hits in a cluster yields a distribution for $\theta_{c,reco}$. The most common entry is taken to be the cherenkov angle for that event.

In practice (and in reality), the interaction vertex could be anywhere in the tank and the particles could go in any direction. 
Since ANNIE is also not a sphere, you get elongation and distortion effects baked into your PMT hit distribution. To simplify things,
we can take 3 PMT hits and create a plane that contains each one. Any 3 points define a circle, so we can create a circle with some radius
and center point that encompasses each point on that plane. We then have, for each 3-hit combination in an event, a reconstructed cherenkov circle
and an associated center and radius.
 
If the interaction vertex was directly above the cherenkov circle center, an opening angle could be easily calculated. Given what was described above,
this is likely not the case for a majority of 3-hit combos, so instead the interaction vertex + cherenkov circle make up an oblique cone 
(again, this is an oversimplification because you expect the cherenkov light distribution emitted from the particle as it travels the medium to be 
symmetric (and circular), but when it hits a cylindrical surface, the hit distribution will be elongated and potentially assymetric. 
We can then use simple trig arguments to solve for the opening angle of the cone; however, there exists two opening angles (smaller and larger) for an
oblique cone, so to remedy this ambiguity we take that projected circle and essentially "pitch" it up to face the interaction vertex, 
basically manufacturing that the vertex + circle center + radius all make up a right triangle. In practice, this seems to work for low energy simulated
events. We can then directly solve for the opening angle of the cone (the reconstructed cherenkov angle). 

We repeat this calculation for all 3-hit PMT combinations, while mandating that all hits must be unique and throwing 
away calculations where 2 or 3 of the PMTs make up a line (this is sometimes the case for barrel PMTs) - this is done 
because you can't construct a circle with 3 points when two or more of the points are in a line. A histogram
(many hits = many combos = distribution of angles) is created containing the reconstructed cherenkov angles for a given event,
with 3-deg bin sizes, we then (following Super-K, although they use 1-deg bin sizes since they have more granularity and more PMTs = more combinations/entries
to include in their histogram) select the most populated bin. Since there may be flucuations or individual bins that overshine a general peak across
multiple bins, we sum up the neighboring bins to the left and right of $\theta_i$ (so 3 bins in total), and select the $\theta_i$ with the highest count
total. This is taken as the reconstructed cherenkov angle.

Since the effects of reflection, scattering, dark noise, etc... can all contribute to muddying the cherenkov cone profile, some hit filtering is utilized. The filtering was carried out trying to optimize the expected angle distributions for low energy gammas (~42 degrees or more), muons (~35 degrees or less), and electrons (~42 degrees). Hit filtering based on timing and charge is utlized; the code searches for cluster hits within 13ns of the first hit. 13ns is the travel time in water of light across the detector (PMT rack specifically, from furthest distanced PMTs). This will hopefully reduce the contribution from reflected light. From there, the top 10 charged hits (based on p.e. values) are selected (to limit computation time and grab the brighest hits of the ring) and the code reconstructs the 3-hit combinations. This seems to reproduce the expected angle distributions the best and gives the best rejection yields for gamma and electrons vs low energy muons.


- Currently designed for WCSim generated events (MC).
- Need the python modules `uproot3`, `tqdm`, and `itertools`.
