from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
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
from generate_card import *
from datetime import datetime, date

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print ('Using '+str(device)+' to run this training')


best_model_dir = '/home/samhuang/ML/best_model/'
#save_model_name = best_model_dir+'best_model_ternary_P-CNN2_event_kappa0.23_fiximag/'
#save_model_name = best_model_dir+'best_model_ternary2_P-CNN_event_kappa0.23_fiximag/'
#save_model_name = best_model_dir+'best_model_senary_P_CNN2_kappa0.2302_fiximag/'
save_model_name = best_model_dir+'best_model_septenary_P_CNN2_kappa0.2305_fiximag/'
os.system('mkdir '+save_model_name)

os.system('mkdir '+save_model_name+'Try/')

sample_folder = '/home/samhuang/ML/sample/event_base/'
#data_folder = sample_folder+"samples_kappa0.23/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj/"
#data_folder = sample_folder+"samples_kappa0.23/VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj/"
#data_folder = sample_folder+"samples_kappa0.2302/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj_and_VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj/"
data_folder = sample_folder+"samples_kappa0.2305/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj_and_VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj_and_proc_ppjjjj/"
#signal=[r'$W^+/W^+$',r'$W^-/W^-$',r'$Z/Z$']
#signal=[r'$W^+/W^-$',r'$W^+/Z$',r'$W^-/Z$']
#signal=[r'$W^+/W^+$',r'$W^-/W^-$',r'$Z/Z$', r'$W^+/W^-$', r'$W^+/Z$', r'$W^-/Z$']
signal=[r'$W^+/W^+$',r'$W^-/W^-$',r'$Z/Z$', r'$W^+/W^-$', r'$W^+/Z$', r'$W^-/Z$','background']



# how many times of training to secure statistical features
Ntry = 10
# Data / training parameters.
train_epochs = 100
batch_size = 128
shuffle_size_tr = 1
patience = 10
min_delta = 0.
learning_rate = 1e-5
dim_image = [[75, 75], [[-5, 5], [-0.8, 0.8]]]
limited_datasize =  000

#// make_Qk_image controls the condition whether make Qk image with the given kappa value during reading data. If Negative, then kappa varaible below is usless, and the true value of kappa in this training will be the one set in extract.py
make_Qk_image = False
kappa = 0.23

cc = card(save_model_name)
cc.save_into_card('hyperparameters', Ntry = Ntry, train_epochs = train_epochs, batch_size = batch_size, learning_rate = learning_rate, patience = patience, min_delta = min_delta)
#
#cc.read_card(save_model_name)

###############

if (make_Qk_image):
    data_tr = data_folder+"train2.npy"
    data_va = data_folder+"valid2.npy"
    data_te = data_folder+"test2.npy"
else:
    data_tr = data_folder+"train.npy"
    data_va = data_folder+"valid.npy"
    data_te = data_folder+"test.npy"   

train_loader, N_label, N_train = get_npyimages(data_tr, kappa, batch_size=batch_size, shuffle_size_tr=shuffle_size_tr, limited_datasize=limited_datasize, do_plot=False, make_Qk_image=make_Qk_image, n_class=len(signal))
valid_loader, aa, N_valid = get_npyimages(data_va, kappa, batch_size=batch_size, shuffle_size_tr=0, limited_datasize=int(limited_datasize/4), make_Qk_image=make_Qk_image, n_class=len(signal))
test_loader, aa, N_test = get_npyimages(data_te, kappa, batch_size=1, shuffle_size_tr=0, limited_datasize=int(limited_datasize/4), make_Qk_image=make_Qk_image, n_class=len(signal))

cc.save_into_card('input information', data_tr = data_tr, data_va = data_va, data_te = data_te, kappa = kappa, limited_datasize=limited_datasize, shuffle_size_tr = shuffle_size_tr, dim_image = dim_image, n_class=len(signal))

#model = models.CNN_torch(dim_image=(batch_size, 2, 75, 75), n_class=6).to(device)
#model.do_dynamic_kappa = False
#print (model)


############### folder execution ################
if (os.path.isdir(save_model_name)):
    os.system("trash "+save_model_name+'Try')

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
sys.stdout = Logger()
sys.stdout.give_model_log_directory(save_model_name)
os.system("mkdir "+save_model_name+'Try')
os.system("mkdir "+save_model_name+'figures')


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
    for epoch in range(train_epochs):
        ML.model.train()
        #print ("layer grad", ML.model[0].weight.grad)
        if (make_Qk_image==False):
            ML.finalized_kappa = kappa
        train_loss, train_acc, outputs = ML.train_each_epoch(epoch, train_loader, train_epochs)

        # VALIDATION
        ML.model.eval()
        val_loss, val_acc = ML.valid_each_epoch(epoch, valid_loader)

        # record training history
        history.save_history_in_this_epoch(train_loss, train_acc, val_loss, val_acc)

        if ML.terminate:
            break
    print (history.history.keys())
    collection_history.append(history)
    
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
    
    if i_try+1 < Ntry:
        del ML, model
        gc.collect()
        th.cuda.empty_cache()
print('Finished Training')

make_dot(outputs, params=dict(list(model.named_parameters()))).render(save_model_name+"figures/model", format="png")

# history
os.system('mkdir '+save_model_name+'figures')
# Plot results curves.
fig = plt.figure(1, figsize=(10, 14))
fig.clf()
histories, fig = plot_training_history(collection_history, fig)
fig.savefig(save_model_name+'figures/loss and accuracy.png', dpi=300)
plt.close()
cc.save_into_card('history', train_loss=[histories[0][0].tolist(), histories[1][0].tolist()])
cc.save_into_card('history', valid_loss=[histories[0][1].tolist(), histories[1][1].tolist()])
cc.save_into_card('history', train_acc=[(histories[0][2]*100).tolist(), (histories[1][2]*100).tolist()])
cc.save_into_card('history', valid_acc=[(histories[0][3]*100).tolist(), (histories[1][3]*100).tolist()])

# roc and auc
fig = plt.figure(figsize=(8,6))
roc_auc_values, fig = plot_roc_curve(collection_predictions, collection_labels, signal, fig)
fig.savefig(save_model_name+'figures/roc_auc.png', dpi=300)
AUC_list = []
for ii in range(len(signal)):
    AUC_list.append('{0} {1:2.2f} +- {2:.4f} %'.format(signal[ii], np.mean(roc_auc_values[:,ii])*100, np.std(roc_auc_values[:,ii]*100)))
cc.save_into_card('AUC', AUC_list=AUC_list)

# acc
collection_accs = calculate_ACC(collection_predictions, collection_labels, signal)
ACC_list = []
for i_class, classi in enumerate(collection_accs):
    print (signal[i_class], "(acc = {:.2f} +- {:.4f} %)".format(np.mean(classi)*100, np.std(classi)*100))
    ACC_list.append("{}  acc = {:.2f} +- {:.4f} %".format(signal[i_class], np.mean(classi)*100, np.std(classi)*100))
cc.save_into_card('ACC', ACC_list=ACC_list)

# test
collection_test = np.array(collection_test)
print ("The summarized testing accuracy = {:.2f} +- {:.4f} %, with the loss = {:.4f} +- {:.6f}".format(np.mean(collection_test[:,1]*100), np.std(collection_test[:,1]*100), np.mean(collection_test[:,0]), np.std(collection_test[:,0])))
cc.save_into_card('testing result', ACC="{:.2f} +- {:.4f} %".format(np.mean(collection_test[:,1]*100), np.std(collection_test[:,1]*100)))
cc.save_into_card('testing result', loss="{:.4f} +- {:.6f}".format(np.mean(collection_test[:,0]), np.std(collection_test[:,0])))

# best model
best_model_tag = np.argmax(collection_test[:,1])
print ("Best performance is derived from Model #{:d}, whose loss = {:.4f} and acc = {:.2f} %".format(best_model_tag, collection_test[best_model_tag,0], collection_test[best_model_tag,1]*100))
cc.save_into_card('best model', best_model_tag=str(best_model_tag), ACC = "{:.4f}".format((collection_test[best_model_tag,1]*100).tolist()), loss = "{:.6f}".format((collection_test[best_model_tag,0]).tolist()))

cc.output_card()
sys.stdout.close()

os.system("./../best_model/organize_model_log.sh "+save_model_name+'latest_run.log')
os.system("cat "+save_model_name+"latest_run.log >> "+save_model_name+data_folder.split('/')[5]+'.log')
