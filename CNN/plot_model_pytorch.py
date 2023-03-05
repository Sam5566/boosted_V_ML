from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
os.chdir('/home/samhuang/ML/CNN/')
sys.path.insert(0, '/home/samhuang/ML')
sys.path.insert(0, '/home/samhuang/ML/sample/event_base')
#from readTFR import *
from matplotlib import pyplot as plt
plt.switch_backend('agg')
# import matplotlib as mpl
# from sklearn.metrics import roc_curve, auc
import models_pytorch as models

import torch as th
from torch.utils.data import TensorDataset, DataLoader
from torchviz import make_dot
import numpy as np
import gc 

# import json
# import pandas as pd
# import random
from tqdm import tqdm
from train_utils import *
# import logging
from print_and_draw import *
import pdb
import tensorflow as tf
from writeTFR import determine_entry
from torchvision import transforms
import torchvision


import matplotlib.pyplot as plt
from matplotlib import colors
import copy

from read_pytorch import *
# from datetime import datetime, date
from pytorch_training import *
from datetime import datetime, date

# from datetime import datetime, date

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print ('Using '+str(device)+' to run this training')


best_model_dir = '/home/samhuang/ML/best_model/'
#save_model_name = best_model_dir+'best_model_ternary_P-CNN2_event_kappa0.23_fiximag/'
#save_model_name = best_model_dir+'best_model_ternary2_P-CNN_event_kappa0.23_fiximag/'
#save_model_name = best_model_dir+'best_model_senary_P_CNN2_kappa0.2301_fiximag/'
save_model_name = best_model_dir+'best_model_septenary_P_CNN2_kappa0.2301_fiximag/'
os.system('mkdir '+save_model_name)

os.system('mkdir '+save_model_name+'Try/')

sample_folder = '/home/samhuang/ML/sample/event_base/'
#data_folder = sample_folder+"samples_kappa0.23/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj/"
#data_folder = sample_folder+"samples_kappa0.23/VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj/"
#data_folder = sample_folder+"samples_kappa0.2301/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj_and_VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj/"
data_folder = sample_folder+"samples_kappa0.2301/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj_and_VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj_and_proc_ppjjjj/"
#signal=[r'$W^+/W^+$',r'$W^-/W^-$',r'$Z/Z$']
#signal=[r'$W^+/W^-$',r'$W^+/Z$',r'$W^-/Z$']
#signal=[r'$W^+/W^+$',r'$W^-/W^-$',r'$Z/Z$', r'$W^+/W^-$', r'$W^+/Z$', r'$W^-/Z$']
signal=[r'$W^+/W^+$',r'$W^-/W^-$',r'$Z/Z$', r'$W^+/W^-$', r'$W^+/Z$', r'$W^-/Z$','background']

extra_information = [r'$m_{jj}$']

# how many times of training to secure statistical features
Ntry = 10
# Data / training parameters.
train_epochs = 100
batch_size = 128
shuffle_size_tr = 1
patience = 10
min_delta = 0.
learning_rate = 1e-6
dim_image = [[75, 75], [[-0.8, 0.8], [-0.8, 0.8]]]
limited_datasize =  000

#// make_Qk_image controls the condition whether make Qk image with the given kappa value during reading data. If Negative, then kappa varaible below is usless, and the true value of kappa in this training will be the one set in extract.py
make_Qk_image = False
kappa = 0.23


###############

if (make_Qk_image):
    data_tr = data_folder+"train2.npy"
    data_va = data_folder+"valid2.npy"
    data_te = data_folder+"test2.npy"
else:
    data_tr = data_folder+"train.npy"
    data_va = data_folder+"valid.npy"
    data_te = data_folder+"test.npy"   

#train_loader, N_label, N_train = get_npyimages(data_tr, kappa, batch_size=batch_size, shuffle_size_tr=shuffle_size_tr, limited_datasize=limited_datasize, do_plot=False, make_Qk_image=make_Qk_image, n_class=len(signal))
#valid_loader, aa, N_valid = get_npyimages(data_va, kappa, batch_size=batch_size, shuffle_size_tr=0, limited_datasize=int(limited_datasize/4), make_Qk_image=make_Qk_image, n_class=len(signal))
test_loader, aa, N_test, extra_test_inputs = get_npyimages(data_te, kappa, batch_size=1, shuffle_size_tr=0, limited_datasize=int(limited_datasize/4), make_Qk_image=make_Qk_image, n_class=len(signal), extra_information=extra_information)

#model = models.CNN_torch(dim_image=(batch_size, 2, 75, 75), n_class=6).to(device)
#model.do_dynamic_kappa = False
#print (model)


#######################  plotting setup  ##########
font = {'size'  : 14} 
fontsize=18
fontsize2=16
legendsize=16
plt.rc('font', **font)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"]="in"
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams["xtick.major.size"] = 6 
plt.rcParams['xtick.major.width'] = 1.5 
plt.rcParams["ytick.major.size"] = 6 
plt.rcParams['ytick.major.width'] = 1.5 

###################################################


print ("")
print ("")
print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print ("New Run")
e = datetime.now()
print ("Current date and time = %s" % e)
#print ("Class: "+signal)
print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#######################  main code  #############################

#exit()
# loss and optimizer
def pt_categorical_crossentropy(pred, label):
    #pred = th.nn.Softmax(dim=1)(pred)
    return th.mean(th.sum(-label * th.log(pred), dim=-1))

    #criterion = criterion.cuda()

collection_history = []
collection_predictions = []
collection_labels = []
collection_test = []
for i_try in range(Ntry):
    model = models.CNN2_torch(dim_image=(batch_size, 2, 75, 75), n_class=len(signal)).to(device)
    model.do_dynamic_kappa = False
    model.make_Qk_image = make_Qk_image
    print (model)
    criterion = pt_categorical_crossentropy
    #criterion = th.nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, eps=1.e-7)
    if th.cuda.is_available():
        print ("Send the model to cuda")
        model = model.cuda()
    ML = train_setup(model, optimizer, criterion, device, save_model_name=save_model_name+'Try/'+str(i_try), patience=patience)
    history = train_history(train_epochs)
    #exit()
    
    ML.test_model = ML.model
    ML.test_model.load_state_dict(th.load(save_model_name+'Try/'+str(i_try)))
    ML.test_model.eval()
    test_loss, test_acc = ML.test(test_loader)

    #test_predictions = np.array(ML.test_predictions)
    #shape = test_predictions.shape
    #print (shape)
    #print (test_predictions)
    #test_predictions = (test_predictions.reshape((N_test, N_label)))
    collection_predictions.append(np.array(ML.test_predictions))

    test_labels = []
    collection_labels.append(np.array(ML.test_labels))

    collection_test.append([test_loss, test_acc/100])

    fig = plt.figure(1, figsize=(10, 14))
    fig.clf()
    fig = draw_resonance_peak(model, collection_labels[-1], collection_predictions[-1], signal, extra_test_inputs, extra_information, fig)
    fig.savefig(save_model_name+'figures/resonance_peak'+str(i_try)+'.png', dpi=300)
    fig.tight_layout()
    plt.close()
    
    if i_try+1 < Ntry:
        del ML, model
        gc.collect()
        th.cuda.empty_cache()
print('Finished Training')


collection_test = np.array(collection_test)
print ("The summarized testing accuracy = {:.2f} +- {:.4f} %, with the loss = {:.4f} +- {:.6f}".format(np.mean(collection_test[:,1]*100), np.std(collection_test[:,1]*100), np.mean(collection_test[:,0]), np.std(collection_test[:,0])))

best_model_tag = np.argmax(collection_test[:,1])
print ("Best performance is derived from Model #{:d}, whose loss = {:.4f} and acc = {:.2f} %".format(best_model_tag, collection_test[best_model_tag,0], collection_test[best_model_tag,1]*100))
