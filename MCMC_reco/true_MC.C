#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include <fstream>
#include <vector>
using namespace std;

// -------------------------------------------------------
// This program takes the MC truth information
// from the BeamCluster Tree and dumps it into a .dat file
// for Python Analysis   |  Author: Steven Doran 3/6/23
// -------------------------------------------------------

// Also serves to extract any information from the TriggerTree

// Dump Contents of a Tree with branches to a .txt or .dat file
void true_MC(){

  	// specify file and tree name
	TFile *f=new TFile("WCSim Data/30MeV/electron_swarm_30MeV.ntuple.root"); // opens the root file
	TTree *tr=(TTree*)f->Get("phaseIITriggerTree"); // creates the TTree object
  	
	// keep in mind this is the "raw" event information, and it has not been clusterized.
	// this is important when comparing event numbers (you will have less clustered events than actual events)

	int e_num;  // event number
	
	// truth information
	double vtX, vtY, vtZ, vtT;  // vertex (x,y,z, initial time (t0))
	double dirX, dirY, dirZ;    // direction vector
	double Energy;              // energy of primary particle
	double track_l;             // track length of primary particle (in the tank/in water)

	tr->SetBranchAddress("eventNumber",&e_num);
	tr->SetBranchAddress("trueVtxX",&vtX);
	tr->SetBranchAddress("trueVtxY",&vtY);
	tr->SetBranchAddress("trueVtxZ",&vtZ);
	tr->SetBranchAddress("trueVtxTime",&vtT);
	tr->SetBranchAddress("trueDirX",&dirX);
	tr->SetBranchAddress("trueDirY",&dirY);
	tr->SetBranchAddress("trueDirZ",&dirZ);
	tr->SetBranchAddress("trueMuonEnergy",&Energy);  // just labeled as Muon, since it was designed for muon analysis
	tr->SetBranchAddress("trueTrackLengthInWater",&track_l);  // measured difference between start and stop vertex of primary particle

	
	Long64_t n = tr->GetEntries();

	// create and open the file where the contents will be dumped
        ofstream myfile;
        myfile.open ("WCSim Data/30MeV/mctruth_electron_swarm_30MeV.dat");
 
        // first line are the column headers (modify if needed)
        myfile << "eventNumber true_vertex_X true_vertex_Y true_vertex_Z true_t0 true_dir_X true_dir_Y true_dir_Z "
        << "true_energy true_track_length\n";
 
   	for (int i=0;i<tr->GetEntries();i++){
     	// loop over the tree
     		tr->GetEntry(i);
     		myfile << e_num << " " << vtX << " " << vtY << " " << vtZ << " "
     		<< vtT << " " << dirX << " " << dirY << " " << dirZ << " "
		<< Energy << " " << track_l << "\n";// << a12 << " "
   	}
   	myfile.close();

cout <<"done" << endl;
}
