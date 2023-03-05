#!/bin/bash
out_directory=$1
#"VBF_H5m_wz_jjjj"
launch_directory=$2
#"event2"

out_file="tag_$3_delphes_events.root"

echo $out_directory

if [ -e $out_file ]; then
    rm $out_file
fi

if [ -e $out_directory/Events/$launch_directory/tag_$3_pythia8_events.hepmc.gz ]; then
    gunzip $out_directory/Events/$launch_directory/tag_$3_pythia8_events.hepmc.gz
fi

DelphesHepMC /home/samhuang/ML/sample/test_jenis_setting/Cards/delphes_card.dat /home/samhuang/ML/sample/$out_directory/Events/$launch_directory/$out_file /home/samhuang/ML/sample/$out_directory/Events/$launch_directory/tag_$3_pythia8_events.hepmc 
