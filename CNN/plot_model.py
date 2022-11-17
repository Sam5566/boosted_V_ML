#from __future__ import absolute_importpython
#from __future__ import division
#from __future__ import print_function

import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import sys
#os.chdir('/home/samhuang/ML/')
sys.path.insert(0, '/home/samhuang/ML')
sys.path.insert(0, '/home/samhuang/ML/sample')
#sys.path.insert(0, '/home/samhuang/../public/Polar_new/samples')
from readTFR_4jet import *
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc
import models
import math
import numpy as np
import tensorflow as tf
import json
import pandas as pd
import random
from tqdm import tqdm
from train_utils import *
import logging
from datetime import datetime, date
from print_and_draw import *
import json


#################  input and variables  #######################
# Data / training parameters.
train_epochs = 500
batch_size = 512
shuffle_size_tr = 0
patience = 10
min_delta = 0.
learning_rate = 1e-4
dim_image = [[75, 75], [[-0.8, 0.8], [-0.8, 0.8]]]
best_model_dir = '/home/samhuang/ML/best_model/'
save_model_name = best_model_dir+'best_model_ternary_CNN_4jetpt_kappa0.15/'
signal=[r'$W^+$',r'$W^-$',r'$Z$']
signal=[r'$W^+/W^+$',r'$W^-/W^-$',r'$Z/Z$', r'$W^+/W^-$', r'$W^+/Z$', r'$W^-/Z$']

# Input datasets
sample_folder = '/home/samhuang/ML/sample/jet_base/'
sample_folder = '/home/samhuang/ML/sample/event_base/'
#data_folder = "sample/samples_kappa0.15/samples_kappa0.15/"
data_folder = sample_folder+"samples_kappa0.15_4jet/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj/"
data_folder = sample_folder+"samples_kappa0.15_4jet/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj_and_VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj/"
#data_folder = "/home/samhuang/../public/Polar_new/samples/"
#data_folder = "samples/"
data_tr = data_folder+"train.tfrecord"
data_vl = data_folder+"valid.tfrecord" 
data_te = data_folder+"test.tfrecord" 

#######################  main code  #############################
collection_predictions = []
collection_labels = []
collection_test = []
for ii in range(10):
    loaded_model = tf.keras.models.load_model(save_model_name+"/Try/"+str(ii))

    dataset_te, te_total  = get_dataset(data_te, repeat=False, 
                                    batch_size=1, 
                                    dim_image=dim_image+[True], 
                                    shuffle=0, N_labels=len(signal))

    #results = loaded_model.evaluate(dataset_te)
    #print("Testing Loss = {0:f}, Testing Accuracy = {1:f}".format(results[0], results[1]))

    labels = [x[1][0].tolist() for x in dataset_te.as_numpy_iterator()]
    collection_labels.append(np.array(labels))

    dataset_te, te_total  = get_dataset(data_te, repeat=False, 
                                    batch_size=1, 
                                    dim_image=dim_image+[False], 
                                    shuffle=0, N_labels=len(signal))

    predictions = loaded_model.predict(dataset_te).tolist()
    #print (predictions)
    collection_predictions.append(np.array(predictions))
    data = {'test_scores': predictions, 'test_labels': labels}


fig = plt.figure(figsize=(8,6))
roc_auc_values, fig = plot_roc_curve(collection_predictions, collection_labels, signal, fig)
fig.savefig(save_model_name+'/figures/roc_auc.png', dpi=300)

fig, ax = plt.subplots(figsize=(10,10))
plot_confusion_matrix(collection_predictions, collection_labels, signal, ax)
fig.savefig(save_model_name+'/figures/confusion_matrix.png', dpi=300)

#collection_test = np.array(collection_test)
#print ("The summarized testing accuracy = %.2f +- %.4f" %(np.mean(collection_test[:,1]*100), np.std(collection_test[:,1]*100)))

'''
labels = np.array(labels)
predictions = np.array(predictions)
n_class = np.shape(labels)[1]
print (n_class)

 
fpr, tpr, roc_auc, si = dict(), dict(), dict(), dict()
for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    si[i] = tpr[i] / np.sqrt(fpr[i])
 
fig = plt.figure(figsize=(8,6))
#signal=[r'$W^+$',r'$W^-$',r'$Z$']
signal=[r'$W^+$',r'$Z$']
for i in range(n_class):
    plt.plot(fpr[i], tpr[i], label='{0} (auc = {1:0.4f})'.format(signal[i], roc_auc[i]))
            
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right', fontsize=15)
fig.savefig(save_model_name+'/figures/roc_auc.png', dpi=300)
plt.close()



# Plot SIC
plt.figure(figsize=(8,6))
for i in range(n_class):
    plt.plot(tpr[i], si[i], label='{0}'.format(signal[i]))
            
plt.xlabel('True Positive Rate')
plt.ylabel('Significance Improvement')
plt.title('SIC')
plt.legend(loc='lower right', fontsize=15)
plt.savefig(save_model_name+'/figures/sic.png', dpi=300)



##############################################################
sys.stdout.close()
'''