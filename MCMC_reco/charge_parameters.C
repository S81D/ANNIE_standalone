#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include <fstream>
#include <vector>
using namespace std;

// -------------------------------------------------------
// This program takes the cluster-level Event information
// (charge parameters) from the BeamCluster Tree and
// dumps it into a .dat file for Python Analysis
// Author: Steven Doran 3/6/23
// -------------------------------------------------------


// Dump Contents of a Tree with branches to a .txt or .dat file
void charge_parameters(){

  	// specify file and tree name
	TFile *f=new TFile("WCSim_Data/30MeV/electron_swarm_30MeV.ntuple.root"); // opens the root file
	TTree *tr=(TTree*)f->Get("phaseIITankClusterTree"); // creates the TTree object
  

	// Event-Level (cluster-level) parameters

	//float ; // create variables of the same type as the branches you want to access
	int e_num;
	double c_t, c_c, c_pe, c_mpe, ccb;
	unsigned int c_h;

	std::vector<int> *hChanMC = 0;

	// for all the TTree branches you need this
	tr->SetBranchAddress("eventNumber",&e_num);
	tr->SetBranchAddress("clusterCharge",&c_c);
	tr->SetBranchAddress("clusterPE",&c_pe);
	tr->SetBranchAddress("clusterMaxPE",&c_mpe);
	tr->SetBranchAddress("clusterChargeBalance",&ccb);
        tr->SetBranchAddress("clusterHits",&c_h);


	//tr->SetBranchAddress("",&);
        //tr->SetBranchAddress("",&);


	// create and open the file where the contents will be dumped
	ofstream myfile;
	myfile.open ("WCSim_Data/30MeV/charge_event_electron_swarm_30MeV.dat");
	
	// first line are the column headers (modify if needed)
	myfile << "eventNumber clusterCharge " 
	<< "clusterPE clusterMaxPE clusterChargeBalance clusterHits\n";

  for (int i=0;i<tr->GetEntries();i++){
    // loop over the tree
    tr->GetEntry(i);
    myfile << e_num << " " << c_c << " " << c_pe << " " << c_mpe << " "
    << ccb << " " << c_h << "\n";// << a12 << " "
  }
  myfile.close();

cout <<"done" << endl;
}
