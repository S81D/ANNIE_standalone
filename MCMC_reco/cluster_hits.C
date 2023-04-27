#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include <fstream>
#include <vector>
using namespace std;

// -------------------------------------------------------
// This program takes the cluster-level hits information
// from the BeamCluster Tree and dumps it into a .dat file
// for Python Analysis   |  Author: Steven Doran 3/6/23
// -------------------------------------------------------


// Dump Contents of a Tree with branches to a .txt or .dat file
void cluster_hits(){

  	// specify file and tree name
	TFile *f=new TFile("WCSim Data/30MeV/electron_swarm_30MeV.ntuple.root"); // opens the root file
	TTree *tr=(TTree*)f->Get("phaseIITankClusterTree"); // creates the TTree object
  	
	// event-level information
	int e_num;

	// Hit-level (for a cluster) text file

	std::vector<int> *hChanMC = 0;
	std::vector<double> *fhitT = 0;
	std::vector<double> *fhitX = 0;
	std::vector<double> *fhitY = 0;
	std::vector<double> *fhitZ = 0;
	std::vector<double> *fhitQ = 0;
	std::vector<double> *fhitPE = 0;
		
	tr->SetBranchAddress("eventNumber",&e_num);
	tr->SetBranchAddress("hitChankeyMC",&hChanMC);
	tr->SetBranchAddress("hitT", &fhitT);
	tr->SetBranchAddress("hitX", &fhitX);
	tr->SetBranchAddress("hitY", &fhitY);
	tr->SetBranchAddress("hitZ", &fhitZ);
	tr->SetBranchAddress("hitQ", &fhitQ);
	tr->SetBranchAddress("hitPE", &fhitPE);

	
	Long64_t n = tr->GetEntries();

	ofstream myfile_hits;
	myfile_hits.open ("WCSim Data/30MeV/cluster_hits_electron_swarm_30MeV.dat");
	myfile_hits << "EventNumber Channel hitT hitX hitY hitZ hitQ hitPE\n";

	for (Long64_t i=0;i<n;i++){
		tr->GetEntry(i);
		ULong_t nsig = fhitT->size();
		double *sT = fhitT->data();
		double *sX = fhitX->data();
		double *sY = fhitY->data();
		double *sZ = fhitZ->data();
		double *sQ = fhitQ->data();
		double *sPE = fhitPE->data();	
		int *sChan = hChanMC->data();
		for (ULong_t j = 0; j < nsig; j++) {
			myfile_hits << e_num << " " << sChan[j] << " " << sT[j] << " "
			<< sX[j] << " " << sY[j] << " " << sZ[j] << " "
			<< sQ[j] << " " << sPE[j] <<  "\n";	
	}		
	
     	}
	myfile_hits.close();
	
cout <<"done" << endl;
}
