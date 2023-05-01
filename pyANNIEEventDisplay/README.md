# pyANNIEEventDisplay

This Event Display shows the ANNIE detector's response to neutrino interactions occuring within the Tank. Designed for neutral current events (low energy, no exiting leptons) so currently there is no display for the FMV or the MRD.

Will eventually be integrated into ToolAnalysis framework.

This downstream python event display takes information extracted from .ROOT files produced using Tools in ToolAnalysis and shows the timing and charge response of the PMTs. Currently designed for MC events (via WCSim), but will be expanded to show real events.

## How to use (for MC)

1. Run the BeamClusterMC ToolChain withinin ToolAnalysis to extract information from the WCSim output .ROOT file. The clusterization tools will filter hits within an alloted time interval (defined as a cluster) and pass these hits along as an event to the PhaseIITreeMaker tool. This tool will create a .ROOT file containing trees (cluster-level and raw hits) and histograms of various parameters of the events.
2. Store the .ROOT output file from ToolAnalysis into the `/MC_Data` folder.
3. Using the .C scripts located in the main directory (`charge_parameters.C`, `cluster_hits.C`, `true_MC.C`), extract hits and cluster information from each event into .dat files to be read by the jupyter notebook event display.
      - For each .C file (3 in total), adjust the filemame passed to the TFiles and the name/path of the output `.dat` file. `charge_parameters.C` reads the event-level information (charge balance, maximum p.e., number of hits, etc..) from the TankClusterTree. `cluster_hits.C` reads the hit-level information (multiple hits per event) (hit time, hit location, p.e.'s, etc..) from the TankClusterTree. `true_MC.C` reads the truth event-level information from the raw TriggerTree. 
      - Run the scripts (3 in total):
```
root -l scriptname.C
```
You will have produced corresponding `.dat` files for each script in the `/Extracted_Data` folder.

4. Run the event display script `python3 pyANNIEEventDisplay.py`. Edit the script as needed. Event Display plots are saved in `2D_Plots`.

##
Notes: 
- `FullTankPMTGeometry.csv` contains the geometry, locations, type, and overall information of the PMTs and is important for the jupyter notebook.
* `\Extracted_Data` contains some example .dat files produced from the .C scripts.
+ The `2D_Plots` directory contains examples of the event display images.
+ There exists a .ipynb script (original) that will display plots in 3D corresponding to the Tank's geometry. This is not included.
+ Unless you want hundreds of photos, do not run this script for all events. Produce a few at a time.
