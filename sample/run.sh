kappa_value=0.15 #// this value need to be modified in main() of File extract.py
extra="" #// this value need to be modified in main() of File extract.py

#python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/Events/run_1/tag_1_delphes_events.root /home/samhuang/ML/sample//test3/Events/run_7/tag_1_delphes_events.root &> extract.log1 &
#python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/Events/run_01/tag_1_delphes_events.root /home/samhuang/ML/sample//test3/Events/run_7/tag_1_delphes_events.root &> extract.log2 &
#python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/Events/run_01/tag_1_delphes_events.root /home/samhuang/ML/sample//test3/Events/run_7/tag_1_delphes_events.root &> extract.log3 &
python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/tmp/tag_1_delphes_events.root &> extract.log1 &
python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/tmp/tag_1_delphes_events.root &> extract.log2 &
python extract.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/tmp/tag_1_delphes_events.root &> extract.log3 &
wait
python plot_jet_sample.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/ /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/ /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/ & 
python3 convert.py samples_kappa$kappa_value$extra/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value$extra/VBF_H5mm_ww_jjjj.npy samples_kappa$kappa_value$extra/VBF_H5z_zz_jjjj.npy &> convert.log1 &
python3 convert2_WmZ.py samples_kappa$kappa_value/VBF_H5mm_ww_jjjj.npy samples_kappa$kappa_value/VBF_H5z_zz_jjjj.npy &> convert.log2 &
python3 convert2_WpZ.py samples_kappa$kappa_value/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value/VBF_H5z_zz_jjjj.npy &> convert.log3 &
python3 convert2_WpWm.py samples_kappa$kappa_value/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value/VBF_H5mm_ww_jjjj.npy &> convert.log4 &
