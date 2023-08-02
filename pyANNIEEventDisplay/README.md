# pyANNIEEventDisplay

This Event Display shows the ANNIE detector's response to neutrino interactions occuring within the Tank. Designed for neutral current events (low energy, no exiting leptons) so currently there is no display for the FMV or the MRD.

Will eventually be integrated into ToolAnalysis framework.

This downstream python event display takes information extracted from .ROOT files produced using Tools in ToolAnalysis and shows the timing and charge response of the PMTs. Currently designed for MC events (via WCSim), but will be expanded to show real events.

## How to use (for MC)

1. Run the BeamClusterMC ToolChain withinin ToolAnalysis to extract information from the WCSim output .ROOT file. The clusterization tools will filter hits within an alloted time interval (defined as a cluster) and pass these hits along as an event to the PhaseIITreeMaker tool. This tool will create a .ROOT file containing trees (cluster-level and raw hits) and histograms of various parameters of the events.
2. Store the .ROOT output file from ToolAnalysis into the `/MC_Data` folder.
3. Run the event display script `python3 pyANNIEEventDisplay.py`. Edit the script as needed. Event Display plots are saved in `2D_Plots`.

##
Notes: 
- `pyANNIEEventDisplay3D.ipynb` is the corresponding 3D event display. Fire up a jupyternotebook to view the events in 3D. Better used on an event-by-event basis, as each event generates 2 interactive windows.
- There is a png file displaying the ANNIE PMT geometry. In the 3D event display, there is an option to disable the PMT's that weren't hit - this makes it much easier to visualize the event and the hit PMTs. This png file is here for reference. 
- `FullTankPMTGeometry.csv` contains the geometry, locations, type, and overall information of the PMTs and is needed for the scripts. Leave it in the main directory.
+ The `2D_Plots` and `3D_Plots` directories contain examples of the event display images.
+ You will need the python module `uproot3` to read from the root files directly.
