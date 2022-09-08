#!/bin/bash
out_directory="VBF_H5z_zz_jjjj1"

out_file="tag_1_delphes_events.root"

echo $out_directory

if [ -e $out_file ]; then
    rm $out_file
fi

if [ -e $out_directory/Events/run_01/tag_1_pythia8_events.hepmc.gz ]; then
    gunzip $out_directory/Events/run_01/tag_1_pythia8_events.hepmc.gz
fi

DelphesHepMC /home/samhuang/ML/sample/test_jenis_setting/Cards/delphes_card.dat /home/samhuang/ML/sample/$out_directory/Events/run_01/$out_file /home/samhuang/ML/sample/$out_directory/Events/run_01/tag_1_pythia8_events.hepmc
