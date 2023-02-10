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

# import json
# import pandas as pd
# import random
from tqdm import tqdm
from train_utils import *
# import logging
from print_and_draw import *
import pdb

import matplotlib.pyplot as plt
from matplotlib import colors
import copy

from read_pytorch import *
# from datetime import datetime, date
from pytorch_training import *
from datetime import datetime, date

##################################################################
# from datetime import datetime, date

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print ('Using '+str(device)+' to run this training')


best_model_dir = '/home/samhuang/ML/best_model/'
save_model_name = best_model_dir+'best_model_ternary_CNN2_kappa0.23_fiximag/'
os.system('mkdir '+save_model_name)
os.system('mkdir '+save_model_name+'Try/')

sample_folder = '/home/samhuang/ML/sample/event_base/'
#sample_folder = '/home/samhuang/Fishbone/'
#data_folder = "sample/samples_kappa0.15/samples_kappa0.15/"
data_folder = sample_folder+"samples_kappa0.23/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj_and_VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj/"
signal=[r'$W^+/W^+$',r'$W^-/W^-$',r'$Z/Z$', r'$W^+/W^-$', r'$W^+/Z$', r'$W^-/Z$']



# how many times of training to secure statistical features
Ntry = 1
# Data / training parameters.
train_epochs = 100
batch_size = 512
shuffle_size_tr = 1
patience = 10
min_delta = 0.
learning_rate = 1e-4
dim_image = [[75, 75], [[-0.8, 0.8], [-0.8, 0.8]]]
limited_datasize =  10

#// make_Qk_image controls the condition whether make Qk image with the given kappa value during reading data. If Negative, then kappa varaible below is usless, and the true value of kappa in this training will be the one set in extract.py
make_Qk_image = False
kappa = 0.23


if (make_Qk_image):
    data_tr = data_folder+"train2.npy"
    data_va = data_folder+"valid2.npy"
    data_te = data_folder+"test2.npy"
else:
    data_tr = data_folder+"train.npy"
    data_va = data_folder+"valid.npy"
    data_te = data_folder+"test.npy"   

train_loader, N_label, N_train = get_npyimages(data_tr, kappa, batch_size=batch_size, shuffle_size_tr=shuffle_size_tr, limited_datasize=limited_datasize, do_plot=False, make_Qk_image=make_Qk_image)
valid_loader, aa, N_valid = get_npyimages(data_va, kappa, batch_size=batch_size, shuffle_size_tr=0, limited_datasize=limited_datasize, make_Qk_image=make_Qk_image)
test_loader, aa, N_test = get_npyimages(data_te, kappa, batch_size=1, shuffle_size_tr=0, limited_datasize=limited_datasize, make_Qk_image=make_Qk_image)

#model = models.CNN_torch(dim_image=(batch_size, 2, 75, 75), n_class=6).to(device)
#model.do_dynamic_kappa = False
#print (model)


collection_predictions = []
collection_labels = []
collection_test = []
for i_try in range(Ntry):
    model = models.CNN2_torch(dim_image=(batch_size, 2, 75, 75), n_class=6).to(device)
    model.do_dynamic_kappa = False
    model.finalized_kappa = th.tensor([kappa])
    if th.cuda.is_available():
        print ("Send the model to cuda")
        model = model.cuda()
    criterion = th.nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, eps=1.e-7)
    ML = train_setup(model, optimizer, criterion, device, save_model_name=save_model_name+'Try/'+str(i_try), patience=patience)
    print ("model check", ML.model.do_dynamic_kappa, ML.model.finalized_kappa)
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


    plot_SmoothGrad(ML.test_model, test_loader, collection_predictions, collection_labels, signal)
