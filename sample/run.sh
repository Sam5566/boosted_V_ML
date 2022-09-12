kappa_value=0.15 #// this value need to be modified in main() of File extract.py
extra="" #// this value need to be modified in main() of File extract.py

#python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/Events/run_1/tag_1_delphes_events.root /home/samhuang/ML/sample//test3/Events/run_7/tag_1_delphes_events.root &> extract.log1 &
#python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/Events/run_01/tag_1_delphes_events.root /home/samhuang/ML/sample//test3/Events/run_7/tag_1_delphes_events.root &> extract.log2 &
#python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/Events/run_01/tag_1_delphes_events.root /home/samhuang/ML/sample//test3/Events/run_7/tag_1_delphes_events.root &> extract.log3 &
python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/Events/run_02/tag_1_delphes_events.root &> extract.log1 &
python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/Events/run_02/tag_1_delphes_events.root &> extract.log2 &
python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/Events/run_02/tag_1_delphes_events.root &> extract.log3 &
#python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/Events/run_01/tag_1_delphes_events.root /home/samhuang/ML/sample/jennis_setting/Events/run01/tag_1_delphes_events.root &> extract.log1 &
#python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/Events/run_01/tag_1_delphes_events.root /home/samhuang/ML/sample/jennis_setting/Events/run01/tag_1_delphes_events.root &> extract.log2 &
#python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/Events/run_01/tag_1_delphes_events.root /home/samhuang/ML/sample/jennis_setting/Events/run01/tag_1_delphes_events.root &> extract.log3 &
#pppython extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/Events/run_01/tag_1_delphes_eventsCMS.root &> extract.log1 &
#pppython extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/Events/run_01/tag_1_delphes_eventsCMS.root &> extract.log2 &
###python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/Events/run_01/tag_1_delphes_eventsCMS.root &> extract.log3 &
wait
echo 'Start ploting the selected samples and seperate them into train, valid, and test dataset...'
python plot_jet_sample.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/ /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/ /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/ & 
python convert.py samples_kappa$kappa_value$extra/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value$extra/VBF_H5mm_ww_jjjj.npy samples_kappa$kappa_value$extra/VBF_H5z_zz_jjjj.npy &> convert.log1 &
python convert2_WmZ.py samples_kappa$kappa_value/VBF_H5mm_ww_jjjj.npy samples_kappa$kappa_value/VBF_H5z_zz_jjjj.npy &> convert.log2 &
python convert2_WpZ.py samples_kappa$kappa_value/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value/VBF_H5z_zz_jjjj.npy &> convert.log3 &
python convert2_WpWm.py samples_kappa$kappa_value/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value/VBF_H5mm_ww_jjjj.npy &> convert.log4 &
