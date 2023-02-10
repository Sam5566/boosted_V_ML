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
from torchviz import make_dot
import gc

# import json
# import pandas as pd
# import random
from tqdm import tqdm
from train_utils import *
# import logging
from print_and_draw import *
import pdb
#from writeTFR import determine_entry
#from torchvision import transforms
#import torchvision
#import torchmetrics
from read_pytorch import *
# from datetime import datetime, date
from pytorch_training import *

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print ('Using '+str(device)+' to run this training')


best_model_dir = '/home/samhuang/ML/best_model/'
save_model_name = best_model_dir+'best_model_ternary_CNN_kappa/'
os.system('mkdir '+save_model_name)
os.system('mkdir '+save_model_name+'Try/')

sample_folder = '/home/samhuang/ML/sample/event_base/'
data_folder = sample_folder+"samples_kappa0.23/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj_and_VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj/"
signal=[r'$W^+/W^+$',r'$W^-/W^-$',r'$Z/Z$', r'$W^+/W^-$', r'$W^+/Z$', r'$W^-/Z$']

data_tr = data_folder+"train2.npy"
data_va = data_folder+"valid2.npy"
data_te = data_folder+"test2.npy"

# how many times of training to secure statistical features
Ntry = 1
# Data / training parameters.
train_epochs = 100
batch_size = 512
shuffle_size_tr = 1
patience = 10
min_delta = 0.
learning_rate = 1e-4
limited_datasize = 0000
dim_image = [[75, 75], [[-np.pi, np.pi], [-5, 5]]]


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

####################################################################################################
train_loader, N_label, N_train = get_npydataset(data_tr, batch_size=batch_size, shuffle_size_tr=shuffle_size_tr, limited_datasize=limited_datasize, n_class=len(signal))
valid_loader, aa, N_valid = get_npydataset(data_va, batch_size=batch_size, shuffle_size_tr=0, limited_datasize=int(limited_datasize/4), n_class=len(signal))
test_loader, aa, N_test = get_npydataset(data_te, batch_size=batch_size, shuffle_size_tr=0, limited_datasize=int(limited_datasize/4), n_class=len(signal))





collection_history = []
collection_predictions = []
collection_labels = []
collection_test = []
for i_try in range(Ntry):
    model = models.CNN_torch(dim_image=(batch_size, 2, 75, 75), n_class=N_label).to(device)
    print (model)
    # loss and optimizer
    criterion = th.nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    if th.cuda.is_available():
        print ("Send the model to cuda")
        model = model.cuda()
    criterion = criterion.cuda()
    ML = train_setup(model, optimizer, criterion, device, save_model_name=save_model_name+'Try/'+str(i_try), patience=patience)
    history = train_history(train_epochs)
    #exit()
    for epoch in range(train_epochs):
        ML.model.train()
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
    gc.collect()
    th.cuda.empty_cache()
print('Finished Training')

make_dot(outputs, params=dict(list(model.named_parameters()))).render(save_model_name+"figures/model", format="png")

os.system('mkdir '+save_model_name+'figures')
# Plot results curves.
fig = plt.figure(1, figsize=(10, 14))
fig.clf()
fig = plot_training_history(collection_history, fig)
fig.savefig(save_model_name+'figures/loss and accuracy.png', dpi=300)
plt.close()

fig = plt.figure(figsize=(8,6))
roc_auc_values, fig = plot_roc_curve(collection_predictions, collection_labels, signal, fig)
fig.savefig(save_model_name+'figures/roc_auc.png', dpi=300)

collection_accs = calculate_ACC(collection_predictions, collection_labels, signal)
for i_class, classi in enumerate(collection_accs):
    print (signal[i_class], "(acc = {:.2f} +- {:.4f} %".format(np.mean(classi)*100, np.std(classi)*100))

collection_test = np.array(collection_test)
print ("The summarized testing accuracy = {:.2f} +- {:.4f} %, with the loss = {:.4f} +- {:.6f}".format(np.mean(collection_test[:,1]*100), np.std(collection_test[:,1]*100), np.mean(collection_test[:,0]), np.std(collection_test[:,0])))



sys.stdout.close()

os.system("./../best_model/organize_model_log.sh "+save_model_name+'latest_run.log')
os.system("cat "+save_model_name+"latest_run.log >> "+save_model_name+data_folder.split('/')[5]+'.log')
