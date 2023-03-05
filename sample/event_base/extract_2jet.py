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

def delta_phi(phi_1, phi_2):
    dPhi = abs(phi_1 - phi_2)
    if dPhi > np.pi:
        return 2 * np.pi - dPhi
    else:
        return dPhi

def match_particle_and_jet(jet_list, particle_list, particle_order_list):
	flat_po_list = np.sum(particle_order_list)
	#print (flat_po_list)
	#print ("#############################")
	#print (flat_po_list.count(3))
	if (flat_po_list.count(1) == 1) and (flat_po_list.count(2) == 1) and (max(flat_po_list)==2):
		#print ("standard case")
		if flat_po_list.index(1) > flat_po_list.index(2):
			jet_list_p = [jet_list[1], jet_list[0]]
			return jet_list_p, particle_list
		return jet_list, particle_list
	elif (flat_po_list.count(1) == 2) and (flat_po_list.count(2) == 1) and (max(flat_po_list)==2):
		print ("Two jets match particle 1")
		if particle_order_list.index([1]) > particle_order_list.index([1,2]):
			jet_list_p = [jet_list[particle_order_list.index([1])], jet_list[particle_order_list.index([1,2])]]
			return jet_list_p, particle_list
		return jet_list, particle_list
	elif (flat_po_list.count(1) == 1) and (flat_po_list.count(2) == 2) and (max(flat_po_list)==2):
		print ("Two jets match particle 1")
		if particle_order_list.index([1,2]) > particle_order_list.index([2]):
			jet_list_p = [jet_list[particle_order_list.index([1,2])], jet_list[particle_order_list.index([2])]]
			return jet_list_p, particle_list
		return jet_list, particle_list
	return jet_list, particle_list

def _shift(a):
	return(a - np.pi*( 2*(a>0)-1 ))

def process(j1, j2, eta_cent, phi_cent, theta, flip):
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

def preprocess2(jet, constituents, kappa):
	s_etaeta, s_etaphi, s_phiphi = 0., 0., 0.
	data = []
	for consti_id, consti in enumerate(constituents):
		try:
			data.append([consti.Phi, consti.Eta, consti.PT, (consti.Charge)*(consti.PT)**kappa/(jet.PT)**kappa])
		except:
			data.append([consti.Phi, consti.Eta, consti.ET,0])
	
	data = np.array(data)
	N_consti = len(data)

	shifted_phi = _shift(data[:,0])
	# shift phi in order to make the data as close as possible if they are around pi boundary
	if (np.var(shifted_phi) < np.var(data[:,0])):
		data[:,0] = shifted_phi
	
	#eta_central = np.sum(data[:,2]*data[:,1])/np.sum(data[:,2])
	#phi_central = np.sum(data[:,2]*data[:,0])/np.sum(data[:,2])
	
	# shift (centralize)
	mu = np.zeros(2)
	for i in range(N_consti):
		mu += data[i,2] * data[i,range(2)]
	mu /= sum(data[:,2])
	#print ("mu",mu)
	data[:,range(0,2)] -= mu
	#print (data[:,range(2)])
	
	# rotation version 2 
	sigma = np.zeros((2,2))
	for ii in range(N_consti):
		sigma += data[ii,2] * np.outer(data[ii,[1,0]], data[ii,[1,0]])
	sigma /= sum(data[:,2])
	#print ("sigma", sigma)
	w, v = np.linalg.eigh(sigma)
	RotMatrix = np.array([v[0,:],v[1,:]])
	#print (RotMatrix)
	for ii in range(N_consti):
		data[ii,range(2)] = RotMatrix @ data[ii,range(2)]

	#print (data[:,range(2)])

    # flip
	lmass, rmass = 0, 0
	umass, dmass = 0, 0
	for i in range(N_consti):
		if data[i,0] > 0:
			rmass += data[i,2]
		else:
			lmass += data[i,2]
		if data[i,1] > 0:
			umass += data[i,2]
		else:
			dmass += data[i,2]
	if lmass > rmass:
		data[:,0] *= -1
	if dmass > umass:
		data[:,1] *= -1
	
	return data[:,2], data[:,1], data[:,0], data[:,3], data[data[:,2]>0.5][:,3]



def preprocess(jet, constituents, kappa):
	pt_sum, eta_central, phi_central = 0., 0., 0.
	s_etaeta, s_etaphi, s_phiphi = 0., 0., 0.
	pt_quadrants = [0., 0., 0., 0.]
	eta_flip, phi_flip = 1., 1.
	pt_news, eta_news, phi_news, Q_kappas, Q_kappas_BDT = [], [], [], [], [] #// BDT variable Q_kappa sum over all constituents that have pT >0.5 GeV

	for consti_id, consti in enumerate(constituents):
		
		try:
			print (consti.Phi, consti.Eta, consti.PT, (consti.Charge)*(consti.PT)**kappa/(jet.PT)**kappa)
			pt_sum += consti.PT
			eta_central += consti.PT * consti.Eta
			phi_central += consti.PT * std_phi(consti.Phi)
			Q_kappas.append((consti.Charge)*(consti.PT)**kappa/(jet.PT)**kappa)
			if consti.PT >0.5:
				Q_kappas_BDT.append((consti.Charge)*(consti.PT)**kappa/(jet.PT)**kappa)
			pt_news.append(consti.PT)
		except:
			print (consti.Phi, consti.Eta, consti.ET,0)
			pt_sum += consti.ET
			eta_central += consti.ET * consti.Eta
			phi_central += consti.ET * std_phi(consti.Phi)
			Q_kappas.append(0.)
			if consti.ET >0.5:
				Q_kappas_BDT.append(0.)
			pt_news.append(consti.ET)
	
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

	print ("sigma",[[s_etaeta, s_etaphi], [s_etaphi, s_phiphi]])
	angle = -np.arctan((-s_etaeta + s_phiphi + np.sqrt((s_etaeta - s_phiphi)**2 + 4. * s_etaphi**2))/(2. * s_etaphi))
	print ([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
	for consti_id, consti in enumerate(constituents):
		eta_shift, phi_shift = consti.Eta - eta_central, std_phi(consti.Phi) - phi_central
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

		print([consti.Phi, consti.Eta], [phi_central, eta_central], '=', [phi_shift, eta_shift])
		print('=>',[phi_shift, eta_shift], '=>', [eta_rotat, phi_rotat])
	if np.argmax(pt_quadrants) == 1:
		phi_flip = -1.
	elif np.argmax(pt_quadrants) == 2:
		phi_flip = -1.
		eta_flip = -1.
	elif np.argmax(pt_quadrants) == 3:
		eta_flip = -1.

	eta_news = [eta_new * eta_flip for eta_new in eta_news]
	phi_news = [phi_new * phi_flip for phi_new in phi_news]

	return pt_news, eta_news, phi_news, Q_kappas, Q_kappas_BDT




def sample_selection(File, histbins, histranges, kappa, signal_label, pbar, tfwriter, imagewriter):
	data_collect = []
	evt_total = 0
	json_list = []
	Npass = np.zeros(5)
	total = File.GetEntries()
	for evt_id, evt in enumerate(File):
		#print ("#########################")
		pTj, Qkj = [], []
		if (evt_id > total):
			break
		particle_number = 2 #// number of the particle in the final state
		particle_list = []
		tmp_particle_list = []

		# new way to find two boson particles (the way shown in jennis's code)
		first_h5pp = True
		id_H5pp, id_w1, id_w2, id_q1, id_q2 = -1, -1, -1, -1, -1
		
		#// find vbf forward jet and vector bosons decay from H5
		for particle_id, particle in enumerate(evt.Particle):
			if (abs(particle.PID) in [255,256,257]):
				if first_h5pp:
					h5pp_mo1 = particle.M1
					h5pp_mo2 = particle.M2
					if ((evt.Particle[particle_id+1].M1 == h5pp_mo1) and (evt.Particle[particle_id+1].M2 == h5pp_mo2)):
						id_q1 = particle_id+1
					else:
						print ('strange q1')
					if ((evt.Particle[particle_id+2].M1 == h5pp_mo1) and (evt.Particle[particle_id+2].M2 == h5pp_mo2)):
						id_q2 = particle_id+2
					else:
						print ('strange q2')
					
					first_h5pp = False
				
				id_H5pp = particle_id

			elif (abs(particle.PID) in [23,24]):
				if ((evt.Particle[id_H5pp].D1 == particle_id) or (particle.M1 == id_w1)):
					id_w1 = particle_id
				if ((evt.Particle[id_H5pp].D2 == particle_id) or (particle.M1 == id_w2)):
					id_w2 = particle_id

		Npass[0] += 1
		particle_list = [evt.Particle[id_w1], evt.Particle[id_w2]]
		particle_pass = [False, False]
		#print (particle_list[0].Status)

		p1, p2 = evt.Particle[particle_list[0].D1], evt.Particle[particle_list[0].D2]
		p3, p4 = evt.Particle[particle_list[1].D1], evt.Particle[particle_list[1].D2]
		
		#if (p1.Eta - p2.Eta)**2 + std_Deltaphi(std_phi(p1.Phi) - std_phi(p2.Phi))**2 < 0.6**2:
		if (p1.Eta - p2.Eta)**2 + delta_phi(p1.Phi, p2.Phi)**2 < 0.6**2:
			particle_pass[0] = True
			#print ("particle1 pass")
		#if (p3.Eta - p4.Eta)**2 + std_Deltaphi(std_phi(p3.Phi) - std_phi(p4.Phi))**2 < 0.6**2:
		if (p3.Eta - p4.Eta)**2 + delta_phi(p3.Phi, p4.Phi)**2 < 0.6**2:
			particle_pass[1] = True
			#print ("particle2 pass")
		if (particle_pass[0]*particle_pass[1]==False): ##need to have both two fat jets
			pbar.update(1)
			continue
		Npass[1] += 1
		#exit()

		Npass[2] += 1

		jet_list = []
		particle_order_list = []
		for jet_id, jet in enumerate(evt.Jet):
			eta_jet, phi_jet = jet.Eta, (jet.Phi)
			#print (evt_id, jet_id, evt.Jet.GetEntries(), eta_jet, jet.PT)
			if (abs(jet.PT-400.) >= 50. or abs(eta_jet) > 1.):
				continue
			#print ("pass3")
			Npass[3] += 1
			particle_order_list.append([])
			[p1, p2] = particle_list
			if (eta_jet - p1.Eta)**2 + delta_phi(phi_jet, p1.Phi)**2 < 0.1**2:
				particle_order_list[-1].append(1)
			if (eta_jet - p2.Eta)**2 + delta_phi(phi_jet, p2.Phi)**2 < 0.1**2: #// \Delta R (V_2,j) < 0.1
				particle_order_list[-1].append(2)
			if (len(particle_order_list[-1])>0):
				jet_list.append(jet)

		particle_order_list.append([]) #//make sure the shape of the list is the same
			
		
		if len(jet_list) ==2 or (len(np.unique(np.sum(particle_order_list)))==2):
			jet_list, particle_list = match_particle_and_jet(jet_list, particle_list, particle_order_list)
		else:
			pbar.update(1)
			continue
		

		Npass[4] += 1

		obs = []
		obs_mass = []
		obs_Qk = []
		N_obs = 2
		for ii in range(len(jet_list)):
			json_obj = {'particle_type': [], 'nodes': [], 'pT': None, 'Qk': None, 'pTj': [], 'Qkj': []}
			obs.append([-1 for x in range(N_obs)])

		
			jet = jet_list[ii]
			constituents = [consti for consti in jet.Constituents if consti != 0]
			pt_news, eta_news, phi_news, Q_kappas, Q_kappas_BDT = preprocess2(jet, constituents, kappa)
		
			for id_1st, consti in enumerate(constituents):
				Rin = np.sqrt((consti.Eta - jet.Eta)**2 + std_Deltaphi(std_phi(consti.Phi) - std_phi(jet.Phi))**2)
				json_obj['nodes'].append([pt_news[id_1st], consti.Eta, std_phi(consti.Phi), eta_news[id_1st], phi_news[id_1st],
				pt_news[id_1st]/jet.PT, Rin, Q_kappas[id_1st]])
 

			eta_list = [x[3] for x in json_obj['nodes']]
			phi_list = [x[4] for x in json_obj['nodes']]
			pT_list  = [x[0] for x in json_obj['nodes']]
			Qk_list  = [x[7] for x in json_obj['nodes']]
			pTj.append(pT_list)
			Qkj.append(Qk_list)
			jet_mass = jet.Mass
			jet_Qk   =	 sum(Q_kappas_BDT)
			obs_mass.append(jet_mass)
			obs_Qk.append(jet_Qk)

		

			obs[-1][0], _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=pT_list)
			obs[-1][1], _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=Qk_list)

			if (ii==0):
				if particle_list[ii].PID==24:
					particle_type='W+'
					json_obj['labels']=[1,0,0]#'W+'
				elif particle_list[ii].PID==-24:
					particle_type='W-'
					json_obj['labels']=[0,1,0]#'W-'
				elif abs(particle_list[ii].PID)==23:
					particle_type='Z'
					json_obj['labels']=[0,0,1]#'Z'
				else:
					print (particle_list[ii].PID)
					print ("no particle type")
			else:
				if particle_list[1].PID==24:
					particle_type+='/W+'
					json_obj['labels']=[1,0,0]#'W+'
				elif particle_list[1].PID==-24:
					particle_type+='/W-'
					json_obj['labels']=[0,1,0]#'W-'
				elif abs(particle_list[1].PID)==23:
					particle_type+='/Z'
					json_obj['labels']=[0,0,1]#'Z'
				else:
					print (particle_list[0].PID)
					print ("no particle type")

		#print (particle_type, np.array(particle_type))
		data_collect.append([particle_type, obs_mass[0], obs_Qk[0], obs_mass[1], obs_Qk[1]])
		image = np.array([np.array(particle_type), obs[0][0], obs[0][1], obs[1][0], obs[1][1]], dtype=object)
			#print (image)
		np.save(imagewriter, image)
		evt_total += 1

		pbar.update(1)#;print (evt_id, len(jet_list));break
		
	return evt_total, data_collect, Npass


def main():
	histbins = [75, 75]
	histranges = [[-0.8, 0.8], [-0.8, 0.8]]
	kappa = float(sys.argv[1])

	inname = sys.argv[2].split('/')[5] #// should be changed with different directory structure
	outputfiledir = sys.argv[2].split('/')[0]+'/'+ sys.argv[2].split('/')[1]+'/'+ sys.argv[2].split('/')[2]+'/'+ sys.argv[2].split('/')[3]+'/' + sys.argv[2].split('/')[4]+'/' + "event_base/samples_kappa"+str(kappa)+'_2jet/'
	os.system('mkdir '+outputfiledir)
	outname = outputfiledir + inname + '.tfrecord'
	imagename = outputfiledir + inname + '.npy'
	countname = outputfiledir + inname + '.count'

	#signal_list = {'VBF_H5pp_ww_jjjj': [1, 0, 0], 'VBF_H5mm_ww_jjjj': [0, 1, 0], 'VBF_H5z_zz_jjjj': [0, 0, 1]}
	signal_list = {'VBF_H5pp_ww_jjjj': [1, 0, 0, 0, 0, 0], 'VBF_H5mm_ww_jjjj': [0, 1, 0, 0, 0, 0], 'VBF_H5z_zz_jjjj': [0, 0, 1, 0, 0, 0], 'VBF_H5z_ww_jjjj': [0, 0, 0, 1, 0, 0], 'VBF_H5p_wz_jjjj': [0, 0, 0, 0, 1, 0], 'VBF_H5m_wz_jjjj': [0, 0, 0, 0, 0, 1]}
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
				#for id, evt in enumerate(chain):
				#	evt_total += function(id,evt, imagewriter, pbar)
				#	if id>= 10**4:
				#		break

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
