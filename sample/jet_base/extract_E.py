from __future__ import absolute_import
import sys, os
os.environ['TERM'] = 'linux'
#import energyflow as ef
import numpy as np
from numpy import pi, sin, cos, sqrt, arctan
import ROOT as r
import json
from scipy.special import expit
from tqdm import tqdm
from itertools import chain
from matplotlib import pyplot as plt
import tensorflow as tf
from tfr_utils import *
from tqdm import tqdm
import pandas as pd

mZ = 91.20

r.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/')
r.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/external/')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/classes/DelphesClasses.h"')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTreeReader.h"')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootConfReader.h"')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTask.h"')
r.gSystem.Load("/usr/local/Delphes-3.4.2/install/lib/libDelphes")

def list_float_feature(features):
	""" Create feature for list of floats"""
	return _list_float_feature(features)

# for tensorflow datasets only
def get_sequence_example_object(data_element_dict):
	""" Creates a SequenceExample object from a dictionary for a single data element 
	data_element_dict is a dictionary for each element in .json file created by the fastjet code. 
	"""
	# Context contains all scalar and list features
	context = tf.train.Features(
			feature=
			{
				'labels' : list_float_feature(data_element_dict['labels']),
				'etal'   : list_float_feature(data_element_dict['etal']),
				'phil'   : list_float_feature(data_element_dict['phil']),
				'ptl'    : list_float_feature(data_element_dict['ptl']),
				'etaj'   : list_float_feature(data_element_dict['etaj']),
				'phij'   : list_float_feature(data_element_dict['phij']),
				'ptj'    : list_float_feature(data_element_dict['ptj']),
			}
	)
	
	# Feature_lists contains all lists of lists
	feature_lists = tf.train.FeatureLists(
			feature_list=None
	#		{
	#
	#		}
	)
				
	sequence_example = tf.train.SequenceExample(context       = context,
												feature_lists = feature_lists)
	
	return sequence_example

def zep(l1, l2, j1, j2):
	z1 = abs(l1.Eta - (j1.Eta + j2.Eta)/2.)/abs(j1.Eta - j2.Eta)
	z2 = abs(l2.Eta - (j1.Eta + j2.Eta)/2.)/abs(j1.Eta - j2.Eta)
	return z1, z2, max(z1, z2)

def minv(p1, p2):
	ptot = p1.P4() + p2.P4()
	return ptot.M()

def std_phi(phi):
	if phi > pi:
		return phi - 2.*pi
	elif phi < -pi:
		return phi + 2.*pi
	else:
		return phi

def std_Deltaphi(Deltaphi):
	if Deltaphi > pi:
		return 2.*pi - Deltaphi
	elif Deltaphi < -pi:
		return 2.*pi + Deltaphi
	else:
		return Deltaphi

def process(j1, j2, eta_cent, phi_cent, theta, flip, histbins, histranges):
	pt_list, eta_list, phi_list = [], [], []
	deques = [j1.Constituents, j2.Constituents]
	for consti in chain(*deques):
		if consti == 0:
			continue
		try:
			pt, eta, phi = consti.PT, consti.Eta, std_phi(consti.Phi)
		except:
			pt, eta, phi = consti.ET, consti.Eta, std_phi(consti.Phi)

		eta, phi = eta - eta_cent, phi - phi_cent
		eta, phi = eta*np.cos(theta) - phi*np.sin(theta), phi*np.cos(theta) + eta*np.sin(theta)
		if flip:
			phi = -phi

		pt_list.append(pt)
		eta_list.append(eta)
		phi_list.append(phi)
	
	return eta_list, phi_list, pt_list

def preprocess(jet, constituents, kappa):
	pt_sum, eta_central, phi_central = 0., 0., 0.
	s_etaeta, s_etaphi, s_phiphi = 0., 0., 0.
	pt_quadrants = [0., 0., 0., 0.]
	eta_flip, phi_flip = 1., 1.
	pt_news, eta_news, phi_news, Q_kappas, E_news = [], [], [], [], []

	for consti_id, consti in enumerate(constituents):
		try:
			pt_sum += consti.PT
			eta_central += consti.PT * consti.Eta
			phi_central += consti.PT * std_phi(consti.Phi)
			Q_kappas.append((consti.Charge)*(consti.PT)**kappa/(jet.PT)**kappa)
			pt_news.append(consti.PT)
			E_news.append(consti.P4().E())
		except:
			pt_sum += consti.ET
			eta_central += consti.ET * consti.Eta
			phi_central += consti.ET * std_phi(consti.Phi)
			Q_kappas.append(0.)
			pt_news.append(consti.ET)
			E_news.append(consti.P4().E())

	eta_central /= pt_sum
	phi_central /= pt_sum

	for consti_id, consti in enumerate(constituents):
		try:
			s_etaeta += consti.PT * (consti.Eta - eta_central)**2
			s_phiphi += consti.PT * (std_phi(consti.Phi) - phi_central)**2
			s_etaphi += consti.PT * (consti.Eta - eta_central) * (std_phi(consti.Phi) - phi_central)
		except:
			s_etaeta += consti.ET * (consti.Eta - eta_central)**2
			s_phiphi += consti.ET * (std_phi(consti.Phi) - phi_central)**2
			s_etaphi += consti.ET * (consti.Eta - eta_central) * (std_phi(consti.Phi) - phi_central)
	
	s_etaeta /= pt_sum
	s_etaphi /= pt_sum
	s_phiphi /= pt_sum

	angle = -np.arctan((-s_etaeta + s_phiphi + np.sqrt((s_etaeta - s_phiphi)**2 + 4. * s_etaphi**2))/(2. * s_etaphi))

	for consti_id, consti in enumerate(constituents):
		eta_shift, phi_shift = consti.Eta - eta_central, std_phi(consti.Phi - phi_central)
		eta_rotat, phi_rotat = eta_shift * np.cos(angle) - phi_shift * np.sin(angle), phi_shift * np.cos(angle) + eta_shift * np.sin(angle)

		eta_news.append(eta_rotat)
		phi_news.append(phi_rotat)

		try:
			if eta_rotat > 0. and phi_rotat > 0.:
				pt_quadrants[0] += consti.PT
			elif eta_rotat > 0. and phi_rotat < 0.:
				pt_quadrants[1] += consti.PT
			elif eta_rotat < 0. and phi_rotat < 0.:
				pt_quadrants[2] += consti.PT
			elif eta_rotat < 0. and phi_rotat > 0.:
				pt_quadrants[3] += consti.PT

		except:
			if eta_rotat > 0. and phi_rotat > 0.:
				pt_quadrants[0] += consti.ET
			elif eta_rotat > 0. and phi_rotat < 0.:
				pt_quadrants[1] += consti.ET
			elif eta_rotat < 0. and phi_rotat < 0.:
				pt_quadrants[2] += consti.ET
			elif eta_rotat < 0. and phi_rotat > 0.:
				pt_quadrants[3] += consti.ET

	if np.argmax(pt_quadrants) == 1:
		phi_flip = -1.
	elif np.argmax(pt_quadrants) == 2:
		phi_flip = -1.
		eta_flip = -1.
	elif np.argmax(pt_quadrants) == 3:
		eta_flip = -1.

	eta_news = [eta_new * eta_flip for eta_new in eta_news]
	phi_news = [phi_new * phi_flip for phi_new in phi_news]

	return pt_news, eta_news, phi_news, Q_kappas, E_news


def sample_selection(File, histbins, histranges, kappa, signal_label, pbar, tfwriter, imagewriter):
	data_collect = []
	evt_total = 0
	json_list = []
	Npass = np.zeros(5)
	total = File.GetEntries()
	for evt_id, evt in enumerate(File):
		pTj, Qkj = [], []
		if (evt_id > total):
			break
		particle_number = 2 #// number of the particle in the final state
		particle_list = []
		tmp_particle_list = []
		for particle_id, particle in enumerate(evt.Particle):
			if (abs(particle.PID) in [255,257]) and (abs(evt.Particle[particle.D1].PID) in [23,24]) and (abs(evt.Particle[particle.D2].PID) in [23,24]):
				if (particle.D1< particle.D2) and (abs(evt.Particle[particle.D1].PID) in [23, 24]) and (abs(evt.Particle[particle.D2].PID) in [23, 24]) and (abs(evt.Particle[evt.Particle[particle.D1].D1].PID) <= 6) and (abs(evt.Particle[evt.Particle[particle.D1].D2].PID) <= 6) and (abs(evt.Particle[evt.Particle[particle.D2].D1].PID) <= 6) and (abs(evt.Particle[evt.Particle[particle.D2].D2].PID) <= 6):
					p1, p2 = evt.Particle[evt.Particle[particle.D1].D1], evt.Particle[evt.Particle[particle.D1].D2]
					p3, p4 = evt.Particle[evt.Particle[particle.D2].D1], evt.Particle[evt.Particle[particle.D2].D2]
					#print ("pass1")
					Npass[0] += 1
					if (p1.Eta - p2.Eta)**2 + std_Deltaphi(std_phi(p1.Phi) - std_phi(p2.Phi))**2 < 0.6**2 and (p3.Eta - p4.Eta)**2 + std_Deltaphi(std_phi(p3.Phi) - std_phi(p4.Phi))**2 < 0.6**2:
						particle_list = [evt.Particle[particle.D1],evt.Particle[particle.D2]]
						#print ("pass2")
						Npass[1] += 1
						break

		if len(particle_list) != 2:
			pbar.update(1)
			continue
		if (particle_list[0].PID==24) and (signal_label!=[1, 0, 0]):
			pbar.update(1)
			continue
		elif (particle_list[0].PID==-24) and (signal_label!=[0, 1, 0]):
			pbar.update(1)
			continue
		elif (particle_list[0].PID==23) and (signal_label!=[0, 0, 1]):
			pbar.update(1)
			continue

		#print (evt_id, p1.Status, p1.M1, p1.M2,  p2.Status, p2.M1, p2.M2)
		Npass[2] += 1

		jet_list = []
		particle_order_list = []
		for jet_id, jet in enumerate(evt.Jet):
			eta_jet, phi_jet = jet.Eta, std_phi(jet.Phi)
			#print (evt_id, jet_id, evt.Jet.GetEntries(), eta_jet, jet.PT)
			if (abs(jet.PT-400.) >= 50. or abs(eta_jet) > 1.):
				continue
			#print ("pass3")
			Npass[3] += 1
			particle_order_list.append([])
			[p1, p2] = particle_list
			#print ((eta_jet - p1.Eta)**2 + std_Deltaphi((phi_jet - std_phi(p1.Phi)))**2, (eta_jet - p2.Eta)**2 + std_Deltaphi((phi_jet - std_phi(p2.Phi)))**2  )
			if (eta_jet - p1.Eta)**2 + std_Deltaphi((phi_jet - std_phi(p1.Phi)))**2 < 0.1**2: #// \Delta R (V_1,j) < 0.1
				particle_order_list[-1].append(1)
			if (eta_jet - p2.Eta)**2 + std_Deltaphi((phi_jet - std_phi(p2.Phi)))**2 < 0.1**2: #// \Delta R (V_2,j) < 0.1
				particle_order_list[-1].append(2)
			if (len(particle_order_list[-1])>0):
				jet_list.append(jet)

		particle_order_list.append([]) #//make sure the shape of the list is the same
				
		#print ("N of jet:", len(jet_list))
		#print (particle_order_list, np.sum(particle_order_list), len(np.unique(np.sum(particle_order_list))), len(jet_list) < 2 or (len(np.unique(np.sum(particle_order_list)))!=2))
		if len(jet_list) < 2 or (len(np.unique(np.sum(particle_order_list)))!=2):
				pbar.update(1)
				continue
		Npass[4] += 1
		#print (evt_id, "found")
		#print (particle_list[0].Eta, particle_list[1].Eta)
		#print (std_phi(particle_list[0].Phi), std_phi(particle_list[1].Phi))
		#print (jet_list[0].Eta, jet_list[1].Eta)
		#print (std_phi(jet_list[0].Phi), std_phi(jet_list[1].Phi))

		json_obj = {'particle_type': [], 'nodes': [], 'pT': None, 'Qk': None, 'E': None, 'pTj': [], 'Qkj': []}


		jet = jet_list[0]
		constituents = [consti for consti in jet.Constituents if consti != 0]
		pt_news, eta_news, phi_news, Q_kappas, E_news = preprocess(jet, constituents, kappa)
		
		for id_1st, consti in enumerate(constituents):
			Rin = np.sqrt((consti.Eta - jet.Eta)**2 + std_Deltaphi(std_phi(consti.Phi) - std_phi(jet.Phi))**2)
			json_obj['nodes'].append([pt_news[id_1st], consti.Eta, std_phi(consti.Phi), eta_news[id_1st], phi_news[id_1st],
			pt_news[id_1st]/jet.PT, Rin, Q_kappas[id_1st], E_news[id_1st]])
 

		eta_list = [x[3] for x in json_obj['nodes']]
		phi_list = [x[4] for x in json_obj['nodes']]
		pT_list  = [x[0] for x in json_obj['nodes']]
		Qk_list  = [x[7] for x in json_obj['nodes']]
		E_list  = [x[8] for x in json_obj['nodes']]
		pTj.append(pT_list)
		Qkj.append(Qk_list)
		jet1_mass = jet.Mass
		jet1_Qk   = sum(Q_kappas)
		
		

		#print ("################################")
		#print (eta_list)
		#print (phi_list)
		#print (pT_list)

		

		hpT1, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=pT_list)
		hQk1, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=Qk_list)
		hE1, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=E_list)

		jet = jet_list[1]
		constituents = [consti for consti in jet.Constituents if consti != 0]
		pt_news, eta_news, phi_news, Q_kappas, E_news = preprocess(jet, constituents, kappa)
		
		for id_1st, consti in enumerate(constituents):
			Rin = np.sqrt((consti.Eta - jet.Eta)**2 + std_Deltaphi(std_phi(consti.Phi) - std_phi(jet.Phi))**2)
			json_obj['nodes'].append([pt_news[id_1st], consti.Eta, std_phi(consti.Phi), eta_news[id_1st], phi_news[id_1st],
			pt_news[id_1st]/jet.PT, Rin, Q_kappas[id_1st], E_news[id_1st]])
 

		eta_list = [x[3] for x in json_obj['nodes']]
		phi_list = [x[4] for x in json_obj['nodes']]
		pT_list  = [x[0] for x in json_obj['nodes']]
		Qk_list  = [x[7] for x in json_obj['nodes']]
		E_list  = [x[8] for x in json_obj['nodes']]
		pTj.append(pT_list)
		Qkj.append(Qk_list)
		jet2_mass = jet.Mass
		jet2_Qk   = sum(Q_kappas)
		json_obj['pTj'] = [item for sublist in pTj for item in sublist]
		json_obj['Qkj'] = [item for sublist in pTj for item in sublist]
		
		if particle_list[0].PID!=particle_list[1].PID:
			print (particle_list[0].PID, particle_list[1].PID)
			print ("particle type of two particles are not the same")
		elif particle_list[0].PID==24:
			json_obj['particle_type']='W+'
			json_obj['labels']=[1,0,0]#'W+'
		elif particle_list[0].PID==-24:
			json_obj['particle_type']='W-'
			json_obj['labels']=[0,1,0]#'W-'
		elif abs(particle_list[0].PID)==23:
			json_obj['particle_type']='Z'
			json_obj['labels']=[0,0,1]#'Z'
		else:
			print (particle_list[0].PID)
			print ("no particle type")
   

		data_collect.append([json_obj['particle_type'], jet1_mass, jet1_Qk])
		data_collect.append([json_obj['particle_type'], jet2_mass, jet2_Qk])

		#print ("################################")
		#print (eta_list)
		#print (phi_list)
		#print (pT_list)

		hpT2, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=pT_list)
		hQk2, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=Qk_list)
		hE2, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=E_list)

		#print ("################################")
		#print (len((hpT1+hpT2)[(hpT1+hpT2)==0]))

		json_obj['pT'] = (hpT1+hpT2)#.tolist()
		json_obj['Qk'] = (hQk1+hQk2)#.tolist()
		json_obj['E'] = (hE1+hE2)#.tolist()
		json_list.append(json_obj)
  
		#sequence_example = get_sequence_example_object(json_obj)
		#print (sequence_example)
		#tfwriter.write(sequence_example.SerializeToString())

		#image = np.array([np.array(json_obj['particle_type']), json_obj['pT'], json_obj['Qk']], dtype=object)
		#image = np.array([np.array(json_obj['particle_type']), hpT1+hpT2, hQk1+hQk2, hE1+hE2], dtype=object)
		image = np.array([np.array(json_obj['particle_type']), hpT1+hpT2, hQk1+hQk2], dtype=object)
		np.save(imagewriter, image)

		evt_total += 1
		pbar.update(1)
		
	return evt_total, data_collect, Npass

			

def main():
	histbins = [75, 75]
	histranges = [[-0.8, 0.8], [-0.8, 0.8]]
	kappa = float(sys.argv[1])

	inname = sys.argv[2].split('/')[5] #// should be changed with different directory structure
	outputfiledir = sys.argv[2].split('/')[0]+'/'+ sys.argv[2].split('/')[1]+'/'+ sys.argv[2].split('/')[2]+'/'+ sys.argv[2].split('/')[3]+'/' + sys.argv[2].split('/')[4]+'/' + "samples_kappa"+str(kappa)+'_E/'
	os.system('mkdir '+outputfiledir)
	outname = outputfiledir + inname + '.tfrecord'
	imagename = outputfiledir + inname + '.npy'
	countname = outputfiledir + inname + '.count'

	signal_list = {'VBF_H5pp_ww_jjjj': [1, 0, 0], 'VBF_H5mm_ww_jjjj': [0, 1, 0], 'VBF_H5z_zz_jjjj': [0, 0, 1]}
	signal_label = signal_list[inname]
	print ("Datatype:",signal_label)

	#create a chain of the Delphes tree
	chain = r.TChain("Delphes")

	for rootfile in sys.argv[2:]:
		chain.Add(rootfile)


	with tqdm(total=chain.GetEntries()) as pbar:
		with tf.io.TFRecordWriter(outname) as tfwriter:
			with open(imagename, 'wb') as imagewriter:
				evt_total = 0
				evt_total, data_collection, Npass = sample_selection(chain, histbins, histranges, kappa, signal_label, pbar, tfwriter, imagewriter)

	with open(countname, 'w+') as f:
		f.write('{0:d}\n'.format(evt_total))

	print (Npass, "out of", chain.GetEntries())

	df = pd.DataFrame([])
	data_collection = np.array(data_collection)
	df['particle_type'] = data_collection[:,0]
	df['jet mass'] = (data_collection[:,1]).astype(float)
	df['jet charge'] = data_collection[:,2].astype(float)
	df.to_csv(outputfiledir + inname + '_properties.txt')
	print ((df))
	print ("Selected data length=%d"%len(df))

if __name__ == '__main__':
	main()
