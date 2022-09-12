#!/bin/bash
out_directory="VBF_H5z_zz_jjjj"
launch_directory="run_02"

out_file="tag_1_delphes_events.root"
out_file2="tag_1_delphes_eventsCMS.root"

echo $out_directory

if [ -e $out_file ]; then
    rm $out_file
fi

if [ -e $out_directory/Events/$launch_directory/tag_1_pythia8_events.hepmc.gz ]; then
    gunzip $out_directory/Events/$launch_directory/tag_1_pythia8_events.hepmc.gz
fi

DelphesHepMC /home/samhuang/ML/sample/test_jenis_setting/Cards/delphes_card.dat /home/samhuang/ML/sample/$out_directory/Events/$launch_directory/$out_file /home/samhuang/ML/sample/$out_directory/Events/$launch_directory/tag_1_pythia8_events.hepmc & > /home/samhuang/ML/sample/$out_directory/Events/$launch_directory/delphes.log1 &
#DelphesHepMC /home/samhuang/ML/sample/test_jenis_setting/Cards/delphes_card_CMS.dat /home/samhuang/ML/sample/$out_directory/Events/$launch_directory/$out_file2 /home/samhuang/ML/sample/$out_directory/Events/$launch_directory/tag_1_pythia8_events.hepmc & > /home/samhuang/ML/sample/$out_directory/Events/$launch_directory/delphes.log2 &
