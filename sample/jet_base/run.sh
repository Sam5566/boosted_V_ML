kappa_value=0.15 #// this value need to be modified in main() of File extract.py
extra="" #// this value need to be modified in main() of File extract.py
extra2=""
#runDir="test2"
#python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/Events/run_02/tag_1_delphes_events.root /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/Events/run_03/tag_1_delphes_events.root &> extract.log4 &
#python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/Events/run_02/tag_1_delphes_events.root /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/Events/run_03/tag_1_delphes_events.root &> extract.log4 &
#python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/Events/run_02/tag_1_delphes_events.root /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/Events/run_03/tag_1_delphes_events.root&> extract.log4 &

python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj_UFO/Events/jet1/tag_1_delphes_events.root /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj_UFO/Events/jet1/tag_2_delphes_events.root &> extract.log4 &
python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj_UFO/Events/jet1/tag_1_delphes_events.root /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj_UFO/Events/jet1/tag_2_delphes_events.root &> extract.log4 &
python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_zz_jjjj_UFO/Events/jet1/tag_1_delphes_events.root /home/samhuang/ML/sample/VBF_H5z_zz_jjjj_UFO/Events/jet1/tag_2_delphes_events.root&> extract.log4 &
wait
echo 'Start ploting the selected samples and seperate them into train, valid, and test dataset...'
python plot_jet_sample.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj_UFO/ /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj_UFO/ /home/samhuang/ML/sample/VBF_H5z_zz_jjjj_UFO/ & 
python convert$extra.py samples_kappa$kappa_value$extra$extra2/VBF_H5pp_ww_jjjj_UFO.npy samples_kappa$kappa_value$extra$extra2/VBF_H5mm_ww_jjjj_UFO.npy samples_kappa$kappa_value$extra$extra2/VBF_H5z_zz_jjjj_UFO.npy &> convert.log1 &
#python convert2_WmZ.py samples_kappa$kappa_value$extra$extra2/VBF_H5mm_ww_jjjj.npy samples_kappa$kappa_value$extra$extra2/VBF_H5z_zz_jjjj.npy &> convert.log2 &
#python convert2_WpZ.py samples_kappa$kappa_value$extra$extra2/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value$extra$extra2/VBF_H5z_zz_jjjj.npy &> convert.log3 &
#python convert2_WpWm.py samples_kappa$kappa_value$extra$extra2/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value$extra$extra2/VBF_H5mm_ww_jjjj.npy &> convert.log4 &


#python convert$extra.py samples_kappa$kappa_value$extra/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value$extra/VBF_H5mm_ww_jjjj.npy samples_kappa$kappa_value$extra/VBF_H5z_zz_jjjj.npy samples_kappa$kappa_value$extra/VBF_H5z_ww_jjjj.npy samples_kappa$kappa_value$extra/VBF_H5p_wz_jjjj.npy samples_kappa$kappa_value$extra/VBF_H5m_wz_jjjj.npy &> convert.log1 &
