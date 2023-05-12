#!/bin/bash

while read energy; do

    input_path="WCSim_Data/${energy}MeV/electron_swarm_${energy}MeV.ntuple.root"

    output_charge="WCSim_Data/${energy}MeV/charge_event_electron_swarm_${energy}MeV.dat"
    output_hits="WCSim_Data/${energy}MeV/cluster_hits_electron_swarm_${energy}MeV.dat"
    output_truth="WCSim_Data/${energy}MeV/truth_electron_swarm_${energy}MeV.dat"

    echo "${energy}"
    
    root << EOF
    .L charge_parameters.C
    charge_parameters("$input_path", "$output_charge");
    .L cluster_hits.C
    cluster_hits("$input_path","$output_hits");
    .L true_MC.C
    true_MC("$input_path","$output_truth")
    .q
EOF


done < "energies.list"
