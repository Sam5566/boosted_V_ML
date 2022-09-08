###############################################################################################
#
# Generate folder for process: VBF > H5ppmm > VV > jjjj
#
###############################################################################################
if [ -d "test4/"  ]
then
	echo "Process already generated."
else
	mg5_aMC << eof
	import model GM_NLO
	generate p p  > H5pp j j , ( H5pp > w+ w+, w+ > j j )
	add process p p  > H5pp~ j j , ( H5pp~ > w- w-, w- > j j )
	add process p p  > H5z j j , ( H5z > z z, z > j j )
	output test4
	exit
eof
fi
###############################################################################################                                                                           #
# Re-generate the Events/ folder
#
###############################################################################################
cd test1/
if [ -d "Events/"  ]
then
	rm -rf Events/
fi
mkdir Events/
## Input of GM model and sampling
## note that the potential couplings is in the form of Logan's paper: ld2<->ld4, ld5->-ld5, Mu1->-Mu1, Mu2->-Mu2
mg5_aMC << eof
launch . -i
launch run_0
#shower=Pythia8
#detector=Delphes
analysis=OFF
0
set tanth 0.226480
set lam2 0.070070
set lam3 -1.331328
set lam4 1.364671
set lam5 -1.963271
set M1coeff 1046.827111
set M2coeff 135.30791
set ebeam1 6500
set ebeam2 6500
set nevents 50000
0
exit
exit
eof

cd ..
