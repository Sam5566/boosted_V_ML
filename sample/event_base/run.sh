kappa_value=0.2305 #// this value need to be modified in main() of File extract.py
extra="" #// this value need to be modified in main() of File extract.py
runDir=""
#mkdir samples_kappa$kappa_value$extra$runDir/
#python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/Events/event6/tag_1_delphes_events.root &> samples_kappa$kappa_value$extra$runDir/extract.log1 &
#python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/Events/event6/tag_1_delphes_events.root &> samples_kappa$kappa_value$extra$runDir/extract.log2 &
#python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/Events/event6/tag_1_delphes_events.root &> samples_kappa$kappa_value$extra$runDir/extract.log3 &
#python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5z_ww_jjjj/Events/event6/tag_1_delphes_events.root  &> samples_kappa$kappa_value$extra$runDir/extract.log4 &
#python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5p_wz_jjjj/Events/event6/tag_1_delphes_events.root  &> samples_kappa$kappa_value$extra$runDir/extract.log5 &
#python extract$extra.py $kappa_value /home/samhuang/ML/sample/VBF_H5m_wz_jjjj/Events/event6/tag_1_delphes_events.root  &> samples_kappa$kappa_value$extra$runDir/extract.log6 &
#wait
echo 'Start ploting the selected samples and seperate them into train, valid, and test dataset...'
#python convert$extra.py samples_kappa$kappa_value$extra$runDir/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5mm_ww_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5z_zz_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5z_ww_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5p_wz_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5m_wz_jjjj.npy &> samples_kappa$kappa_value$extra$runDir/convert.log1 &
#python convert$extra.py samples_kappa$kappa_value$extra$runDir/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5mm_ww_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5z_zz_jjjj.npy &> samples_kappa$kappa_value$extra$runDir/convert.log2 &
#python convert$extra.py samples_kappa$kappa_value$extra$runDir/VBF_H5z_ww_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5p_wz_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5m_wz_jjjj.npy &> samples_kappa$kappa_value$extra$runDir/convert.log3 &
#wait
python convert$extra.py samples_kappa$kappa_value$extra$runDir/VBF_H5pp_ww_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5mm_ww_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5z_zz_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5z_ww_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5p_wz_jjjj.npy samples_kappa$kappa_value$extra$runDir/VBF_H5m_wz_jjjj.npy samples_kappa$kappa_value$extra$runDir/proc_ppjjjj.npy &> samples_kappa$kappa_value$extra$runDir/convert.log4 &
#mv extract.log* convert.log* samples_kappa$kappa_value$extra$runDir/.
#python plot_jet_sample.py $kappa_value /home/samhuang/ML/sample/VBF_H5pp_ww_jjjj/ /home/samhuang/ML/sample/VBF_H5mm_ww_jjjj/ /home/samhuang/ML/sample/VBF_H5z_zz_jjjj/ /home/samhuang/ML/sample/VBF_H5z_ww_jjjj/ /home/samhuang/ML/sample/VBF_H5p_wz_jjjj/ /home/samhuang/ML/sample/VBF_H5m_wz_jjjj/ 
