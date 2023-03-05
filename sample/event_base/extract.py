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
#import tensorflow as tf
from tfr_utils import *
from tqdm import tqdm
import pandas as pd
import cv2 	
from writeTFR import determine_entry

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

def dijet_inv_mass(*arg):
    e_tot, px_tot, py_tot, pz_tot = 0, 0, 0, 0
    
    for jet in arg:
        pt, eta, phi, m = jet[0], jet[1], jet[2], jet[3]
        
        px, py, pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)
        e = np.sqrt(m**2 + px**2 + py**2 + pz**2)
        
        px_tot += px
        py_tot += py
        pz_tot += pz
        e_tot += e
    
    return np.sqrt(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2)

def extract_information(jets):
	#print (dir(jets[0]))
	mjj = dijet_inv_mass([jets[0].PT, jets[0].Eta, jets[0].Phi, jets[0].Mass], [jets[1].PT, jets[1].Eta, jets[1].Phi, jets[1].Mass])

	return [mjj]

def preprocess2(jet_list, constituents_list, kappa):
	data = []
	for jet_id, jet in enumerate(jet_list):
		for consti_id, consti in enumerate(constituents_list[jet_id]):
			try:
				data.append([consti.Phi, consti.Eta, consti.PT, (consti.Charge)*(consti.PT)**kappa/(jet.PT)**kappa, consti.Charge])
			except:
				data.append([consti.Phi, consti.Eta, consti.ET,0, 0])
	
	data = np.array(data)
	N_consti = len(data)

	shifted_phi = _shift(data[:,0])
	# shift phi in order to make the data as close as possible if they are around pi boundary
	if (np.var(shifted_phi) < np.var(data[:,0])):
		data[:,0] = shifted_phi
	
	# shift (centralize) only phi direction
	mu = np.zeros(1)
	for i in range(N_consti):
		mu += data[i,2] * data[i,0]
	mu /= sum(data[:,2])
	#print ("mu",mu)
	data[:,range(0,0)] -= mu

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
	
	return data[:,2], data[:,1], data[:,0], data[:,3], data[data[:,2]>0.5][:,3], data[:,4]

def preprocess(jet_list, constituents_list, kappa): # not using
	pt_sum, eta_central, phi_central = 0., 0., 0.
	s_etaeta, s_etaphi, s_phiphi = 0., 0., 0.
	pt_quadrants = [0., 0., 0., 0.]
	eta_flip, phi_flip = 1., 1.
	pt_news, eta_news, phi_news, Q_kappas, E_news, Q = [], [], [], [], [], []
	for jet_id, jet in enumerate(jet_list):
		for consti_id, consti in enumerate(constituents_list[jet_id]):
			try:
				pt_sum += consti.PT
				eta_central += consti.PT * consti.Eta
				phi_central += consti.PT * std_phi(consti.Phi)
				Q_kappas.append((consti.Charge)*(consti.PT)**kappa/(jet.PT)**kappa)
				pt_news.append(consti.PT)
				E_news.append(consti.P4().E())
				Q.append(consti.Charge)
			except:
				pt_sum += consti.ET
				eta_central += consti.ET * consti.Eta
				phi_central += consti.ET * std_phi(consti.Phi)
				Q_kappas.append(0.)
				pt_news.append(consti.ET)
				E_news.append(consti.P4().E())
				Q.append(0)

	eta_central /= pt_sum
	phi_central /= pt_sum

	for jet_id, jet in enumerate(jet_list):
		for consti_id, consti in enumerate(constituents_list[jet_id]):
			eta_shift, phi_shift = consti.Eta, std_phi(consti.Phi - phi_central) #// no translation needed for eta axis
			#eta_rotat, phi_rotat = eta_shift * np.cos(angle) - phi_shift * np.sin(angle), phi_shift * np.cos(angle) + eta_shift * np.sin(angle)

			eta_news.append(eta_shift)
			phi_news.append(phi_shift)

			try:
				if eta_shift > 0. and phi_shift > 0.:
					pt_quadrants[0] += consti.PT
				elif eta_shift > 0. and phi_shift < 0.:
					pt_quadrants[1] += consti.PT
				elif eta_shift < 0. and phi_shift < 0.:
					pt_quadrants[2] += consti.PT
				elif eta_shift < 0. and phi_shift > 0.:
					pt_quadrants[3] += consti.PT

			except:
				if eta_shift > 0. and phi_shift > 0.:
					pt_quadrants[0] += consti.ET
				elif eta_shift > 0. and phi_shift < 0.:
					pt_quadrants[1] += consti.ET
				elif eta_shift < 0. and phi_shift < 0.:
					pt_quadrants[2] += consti.ET
				elif eta_shift < 0. and phi_shift > 0.:
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

	return pt_news, eta_news, phi_news, Q_kappas, E_news, Q

def sample_selection(evt_id, total, evt, Npass, N_matching, pbar):
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

		elif (abs(particle.PID) in [1, 2, 3, 4, 5, 6]):
			if (first_h5pp):
				continue
			if ((evt.Particle[h5pp_mo1].D1 == particle_id) or (particle.M1 == id_q1)):
				id_q1 = particle_id
			if ((evt.Particle[h5pp_mo2].D2 == particle_id) or (particle.M1 == id_q2)):
				id_q2 = particle_id

	Npass[0] += 1
	if (id_w1 + id_w2)<0:
		print ("Error")
		AssertionError("Does not found end vector boson")
		pbar.update(1)
		return -1, 0, Npass, N_matching
	
	particle_list = [evt.Particle[id_w1], evt.Particle[id_w2]]
	particle_pass = [False, False]
	#print (particle_list[0].Status)
	#print (particle_list[0].Print())
	p1, p2 = evt.Particle[particle_list[0].D1], evt.Particle[particle_list[0].D2]
	p3, p4 = evt.Particle[particle_list[1].D1], evt.Particle[particle_list[1].D2]
	#print ((p1.Eta - p2.Eta)**2 + std_Deltaphi(std_phi(p1.Phi) - std_phi(p2.Phi))**2)
	#print ((p3.Eta - p4.Eta)**2 + std_Deltaphi(std_phi(p3.Phi) - std_phi(p4.Phi))**2)
	
	#if (p1.Eta - p2.Eta)**2 + std_Deltaphi(std_phi(p1.Phi) - std_phi(p2.Phi))**2 < 0.6**2:
	if (p1.Eta - p2.Eta)**2 + delta_phi(p1.Phi, p2.Phi)**2 < 0.6**2:
		particle_pass[0] = True
		N_matching[0] += 1
		#print ("particle1 pass")
	#if (p3.Eta - p4.Eta)**2 + std_Deltaphi(std_phi(p3.Phi) - std_phi(p4.Phi))**2 < 0.6**2:
	if (p3.Eta - p4.Eta)**2 + delta_phi(p3.Phi, p4.Phi)**2 < 0.6**2:
		particle_pass[1] = True
		N_matching[1] += 1
		#print ("particle2 pass")
	if (particle_pass[0]*particle_pass[1]==False):
		pbar.update(1)
		return -1, 0, Npass, N_matching
	Npass[1] += 1

	if len(particle_list) != 2:
		pbar.update(1)
		return -1, 0, Npass, N_matching

	Npass[2] += 1

	jet_list = []
	jet_id_list = []
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
		if (eta_jet - p1.Eta)**2 + delta_phi(phi_jet, p1.Phi)**2 < 0.1**2: #// \Delta R (V_1,j) < 0.1
			particle_order_list[-1].append(1)
		if (eta_jet - p2.Eta)**2 + delta_phi(phi_jet, p2.Phi)**2 < 0.1**2: #// \Delta R (V_2,j) < 0.1
			particle_order_list[-1].append(2)
		if (len(particle_order_list[-1])>0):
			jet_list.append(jet)
			jet_id_list.append(jet_id)

	particle_order_list.append([]) #//make sure the shape of the list is the same
			
	#print ("N of jet:", len(jet_list))
	#print (particle_order_list, np.sum(particle_order_list), len(np.unique(np.sum(particle_order_list))), len(jet_list) < 2 or (len(np.unique(np.sum(particle_order_list)))!=2))
	if len(jet_list) < 2 or (len(np.unique(np.sum(particle_order_list)))!=2):
		pbar.update(1)
		return -1, 0, Npass, N_matching
	if len(jet_list)>2:
		print ("warning:", jet_list)
	jet_list, particle_list = match_particle_and_jet(jet_list, particle_list, particle_order_list)

	Nforward_jet = 0
	forward_jet_list = []
	forward_jet_match = [False, False]
	for jet_id, jet in enumerate(evt.Jet):
		if (jet_id in jet_id_list):
			continue
		eta_jet, phi_jet = jet.Eta, std_phi(jet.Phi)
		#print (evt_id, jet_id, evt.Jet.GetEntries(), eta_jet, jet.PT)
		if (jet.PT >= 30.):
			#print ("jet:",eta_jet, phi_jet)
			#print ("particle",evt.Particle[id_q1].Eta, evt.Particle[id_q1].Phi)
			if (eta_jet - evt.Particle[id_q1].Eta)**2 + delta_phi(phi_jet, evt.Particle[id_q1].Phi)**2 < 0.3**2: #// \Delta R (q_1,j) < 0.3
				forward_jet_match[0] = True
				#print (forward_jet_match)
			elif (eta_jet - evt.Particle[id_q2].Eta)**2 + delta_phi(phi_jet, evt.Particle[id_q2].Phi)**2 < 0.3**2: #// \Delta R (q_2,j) < 0.3
				forward_jet_match[1] = True
				#print (forward_jet_match)
			else:
				continue
			
			forward_jet_list.append(jet)
			Nforward_jet += 1
		if (Nforward_jet >= 2) and (forward_jet_match[1]*forward_jet_match[0]):
			for jet1_id, jet1 in enumerate(forward_jet_list):
				for jet2_id, jet2 in enumerate(forward_jet_list):
					if (jet1_id <= jet2_id):
						continue
					#print (jet1.Eta, jet2.Eta)
					if (abs(jet1.Eta-jet2.Eta)>2):
					#if (abs(jet1.Eta-jet2.Eta)>4):
					#if (abs(jet1.Eta)>2 and abs(jet2.Eta)>2 and (jet1.Eta*jet2.Eta<0):
						jet_list.extend([jet1, jet2])
						#print ("Number of forward jets:", Nforward_jet)
						break
	
	if len(jet_list) < 4:
		pbar.update(1)
		return -1, 0, Npass, N_matching
	if abs(jet_list[-2].Eta-jet_list[-1].Eta)<2:
		pbar.update(1)
		return -1, 0, Npass, N_matching

	Npass[4] += 1
	#print (evt_id, "found")
	#print (particle_list[0].Eta, particle_list[1].Eta)
	#print (std_phi(particle_list[0].Phi), std_phi(particle_list[1].Phi))
	#print (jet_list[0].Eta, jet_list[1].Eta)
	#print (std_phi(jet_list[0].Phi), std_phi(jet_list[1].Phi))
	return particle_list, jet_list, Npass, N_matching

def sample_generation(File, histbins, histranges, kappa, signal_label, pbar, imagewriter, imagewriter2, histbins_true):
	data_collect = []
	evt_total = 0
	json_list = []
	Npass = np.zeros(5)
	N_matching = np.zeros(2)
	total = File.GetEntries()
	for evt_id, evt in enumerate(File):
		pTj, Qkj = [], []
		if (evt_id > total):
			break
		if (evt_id) % 1e6 == 0: #// The purpose of this line is to printout in the next line
			print("")

		if signal_label == 0: #background
			jet_list = []
			particle_list = [-101, -101]
			for jet_id, jet in enumerate(evt.Jet):
				jet_list.append(jet)
				#eta_jet, phi_jet = jet.Eta, std_phi(jet.Phi)
		else: #signal
			particle_list, jet_list, Npass, N_matching = sample_selection(evt_id, total, evt, Npass, N_matching, pbar)
			if type(particle_list) != int:
				pass
			elif particle_list == -1: # does not  pass selection criteria
				continue


		extra_information_list = extract_information(jet_list)
		json_obj = {'particle_type': [], 'nodes': [], 'pT': None, 'Qk': None, 'E': None, 'pTj': [], 'Qkj': []}

		constituents_list = []
		cjets_list = []
		for jet_id, jet in enumerate(jet_list):
			constituents_list.append([consti for consti in jet.Constituents if consti != 0])
			#print (len(constituents_list[-1]))
			#cjets_list.append([jet]*len(constituents_list[jet_id]))
		#constituents_lists = [consti for consti_list in constituents_list for consti in consti_list]
		#print (np.shape(constituents_list))
		#cjets_list = [cjet for cjets in cjets_list for cjet in cjets] # cjets_list is a list of jet that correspond to every constituents in constituents_list

		pt_news, eta_news, phi_news, Q_kappas, Q_kappas_BDT, Q = preprocess2(jet_list, constituents_list, kappa)

		#print (np.shape(pt_news), np.shape(eta_news), np.shape(phi_news), np.shape(Q_kappas))
		id_consti = 0
		for jet_id, jet in enumerate(jet_list):
			for id_1st, consti in enumerate(constituents_list[jet_id]):
				#print (id_consti)
				Rin = np.sqrt((consti.Eta - jet.Eta)**2 + std_Deltaphi(std_phi(consti.Phi) - std_phi(jet.Phi))**2)
				json_obj['nodes'].append([pt_news[id_consti], consti.Eta, std_phi(consti.Phi), eta_news[id_consti], phi_news[id_consti],
				pt_news[id_consti]/jet.PT, Rin, Q_kappas[id_consti], Q[id_consti]])
				id_consti += 1
 

		eta_list = [x[3] for x in json_obj['nodes']]
		phi_list = [x[4] for x in json_obj['nodes']]
		pT_list  = [x[0] for x in json_obj['nodes']]
		Qk_list  = [x[7] for x in json_obj['nodes']]
		pTnorm_list  = [x[5] for x in json_obj['nodes']]
		Q_list  = [x[8] for x in json_obj['nodes']]
		pTj.append(pT_list)
		Qkj.append(Qk_list)
		jet_mass = jet.Mass
		jet_Qk   = sum(Q_kappas_BDT)
		
		json_obj['pTj'] = [item for sublist in pTj for item in sublist]
		json_obj['Qkj'] = [item for sublist in pTj for item in sublist]
		
		#if particle_list[0].PID!=particle_list[1].PID:
		#	print (particle_list[0].PID, particle_list[1].PID)
		#	print ("particle type of two particles are not the same")
		if particle_list[0]==-101: # This is background, no particle list
			json_obj['particle_type']='Background'
			json_obj['labels']=[0,0,0]#'Background'
		elif particle_list[0].PID==24:
			json_obj['particle_type']='W+'
			json_obj['labels']=[1,0,0]#'W+'
		elif particle_list[0].PID==-24:
			json_obj['particle_type']='W-'
			json_obj['labels']=[0,1,0]#'W-'
		elif abs(particle_list[0].PID)==23:
			json_obj['particle_type']='Z'
			json_obj['labels']=[0,0,1]#'Z'
		else: # This should be error
			print (particle_list[0].PID)
			print ("no particle type")
   

		#data_collect.append([json_obj['particle_type'], jet1_mass, jet1_Qk])
		if particle_list[0]==-101: # This is background, no particle list
			json_obj['particle_type']='Background'
			json_obj['labels']=[0,0,0]#'Background'
		elif particle_list[1].PID==24:
			json_obj['particle_type']+='/W+'
			json_obj['labels']=[1,0,0]#'W+'
		elif particle_list[1].PID==-24:
			json_obj['particle_type']+='/W-'
			json_obj['labels']=[0,1,0]#'W-'
		elif abs(particle_list[1].PID)==23:
			json_obj['particle_type']+='/Z'
			json_obj['labels']=[0,0,1]#'Z'
		else: # This should be error
			print (particle_list[0].PID)
			print ("no particle type")
		data_collect.append([json_obj['particle_type'], jet_mass, jet_Qk])

		hpT, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=pT_list)
		hQk, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=Qk_list)
		if (np.isnan(histbins_true[0])):
				hpT, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=pT_list)
				hQk, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins, weights=Qk_list)
		else:        
			hpT, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins_true, weights=pT_list)
			hQk, _, _ = np.histogram2d(eta_list, phi_list, range=histranges, bins=histbins_true, weights=Qk_list)
			hpT = cv2.resize(hpT, histbins, interpolation = cv2.INTER_NEAREST)
			hQk = cv2.resize(hQk, histbins, interpolation = cv2.INTER_NEAREST)

		#print ("################################")
		#print (len((hpT1+hpT2)[(hpT1+hpT2)==0]))

		json_obj['pT'] = (hpT)#.tolist()
		json_obj['Qk'] = (hQk)#.tolist()
		json_list.append(json_obj)
  
		#sequence_example = get_sequence_example_object(json_obj)
		#print (sequence_example)
		#tfwriter.write(sequence_example.SerializeToString())

		#image = np.array([np.array(json_obj['particle_type']), json_obj['pT'], json_obj['Qk']], dtype=object)
		#image = np.array([np.array(json_obj['particle_type']), hpT1+hpT2, hQk1+hQk2, hE1+hE2], dtype=object)
		# This is the image archive 
		image = np.array([np.array(json_obj['particle_type']), hpT, hQk, extra_information_list[0]], dtype=object)
		print (image)
		np.save(imagewriter, image)

		# This is the value of charge and normalized pT of constituents of every considered jets (kappa value is not set yet. Will be calculated during training)
		#print (determine_entry([np.array(json_obj['particle_type'])], evt_total))

		Q_list = np.array(Q_list)
		eta_list = np.array(eta_list)[Q_list!=0]
		phi_list = np.array(phi_list)[Q_list!=0]
		pTnorm_list = np.array(pTnorm_list)[Q_list!=0]
		Q_list = Q_list[Q_list!=0]

		location_list = np.zeros((len(eta_list), 2))
		location_list[:,0] = eta_list
		location_list[:,1] = phi_list

		hQ_list = np.zeros((len(Q_list), histbins[0], histbins[1]))
		for jjj in range(len(Q_list)):
			hQ_list[jjj], _, _ = np.histogram2d([eta_list[jjj]], [phi_list[jjj]], range=histranges, bins=histbins, weights=[Q_list[jjj]])

		image2 = np.array([determine_entry([np.array(json_obj['particle_type'])], evt_total), hpT, hQ_list, pTnorm_list], dtype=object)
		np.save(imagewriter2, image2)

		evt_total += 1
		#print ("central jets' PID:", [particle_list[0].PID, particle_list[1].PID])
		#	print ("forward jets' PID:", [evt.Particle[id_q1].PID, evt.Particle[id_q2].PID])
		pbar.set_description(f'N of data = {evt_total:8d} (passing rate = {evt_total/evt_id*100:.2f}%)')
		pbar.set_postfix({'N matching': N_matching, 'N pass': Npass})
		pbar.update(1)

		##test
		#hQk_test = np.sum(hQ_list*pTnorm_list.reshape(len(pTnorm_list),1,1)**kappa, axis=0)
		#print (np.max((hQk_test-1.e-10)/(hQk+1.e-10)), np.min((hQk_test-1.e-10)/(hQk+1.e-10)))
		#np.set_printoptions(threshold=sys.maxsize)
		#aaaaa = (hQk_test)-(hQk)
		#print (np.where(aaaaa>0, 'X', False))
		#print (np.where(aaaaa>1.e-10, 'X', False))
		#break
		
	return evt_total, data_collect, [Npass, N_matching]

			

def main():
	resolution_false = False
	resolution_scale = 1 # Should be 1 if the histbins setting in the next line is going to be the final histbins and without the need of histbins_true
	histbins = [75, 75]
	if resolution_false == True: # The image's pixel will be histbins, but the image will be first generated in histbins_true then be scaled to histbins
		histbins_true = [int(histbins[0]/resolution_scale), int(histbins[1]/resolution_scale)]
	else: # The image will be generated in scaled histbins.
		histbins_true = [np.nan,np.nan] 
		histbins = [int(histbins[0]/resolution_scale), int(histbins[1]/resolution_scale)]
	histranges = [[-5, 5], [-np.pi, np.pi]]
	kappa = float(sys.argv[1])

	inname = sys.argv[2].split('/')[5] #// should be changed with different directory structure
	outputfiledir = sys.argv[2].split('/')[0]+'/'+ sys.argv[2].split('/')[1]+'/'+ sys.argv[2].split('/')[2]+'/'+ sys.argv[2].split('/')[3]+'/' + sys.argv[2].split('/')[4]+'/' + "event_base/samples_kappa"+str(kappa)+'/'
	os.system('mkdir '+outputfiledir)
	outname = outputfiledir + inname + '.tfrecord'
	imagename = outputfiledir + inname + '.npy'
	imagename2 = outputfiledir + inname + '2.npy'
	countname = outputfiledir + inname + '.count'

	signal_list = {'VBF_H5pp_ww_jjjj': [1, 0, 0], 'VBF_H5mm_ww_jjjj': [0, 1, 0], 'VBF_H5z_zz_jjjj': [0, 0, 1]}
	signal_list = {'VBF_H5pp_ww_jjjj': [1, 0, 0, 0, 0, 0], 'VBF_H5mm_ww_jjjj': [0, 1, 0, 0, 0, 0], 'VBF_H5z_zz_jjjj': [0, 0, 1, 0, 0, 0], 'VBF_H5z_ww_jjjj': [0, 0, 0, 1, 0, 0], 'VBF_H5p_wz_jjjj': [0, 0, 0, 0, 1, 0], 'VBF_H5m_wz_jjjj': [0, 0, 0, 0, 0, 1]}
	try:
		signal_label = signal_list[inname]
		print ("Datatype:",signal_label)
	except:
		print ("This signal does not appear in the default list")
		print ("Take this data as background, and project all the jets onto the jet image.")
		signal_label = 0
		

	#create a chain of the Delphes tree
	chain = r.TChain("Delphes")

	for rootfile in sys.argv[2:]:
		chain.Add(rootfile)


	with tqdm(total=chain.GetEntries(), ncols=170) as pbar:
		with open(imagename, 'wb') as imagewriter:
			with open(imagename2, 'wb') as imagewriter2:
				evt_total = 0
				evt_total, data_collection, N = sample_generation(chain, histbins, histranges, kappa, signal_label, pbar, imagewriter, imagewriter2, histbins_true)

	with open(countname, 'w+') as f:
		f.write('{0:d}\n'.format(evt_total))
	
	for Npass in N:
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
