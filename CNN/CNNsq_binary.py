from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
sys.path.insert(0, '/home/samhuang/ML')
sys.path.insert(0, '/home/samhuang/ML/sample')
#sys.path.insert(0, '/home/samhuang/../public/Polar_new/samples')
from readTFR import *
from matplotlib import pyplot as plt
plt.switch_backend('agg')
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

os.environ['CUDA_VISIBLE_DEVICES']='1'
physical_gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_gpus[0], True)
#################  input and variables  #######################
# Data / training parameters.
train_epochs = 500
batch_size = 512
shuffle_size_tr = 1
patience = 20
min_delta = 0.
learning_rate = 1e-4
N_labels = 2
dim_image = [[75, 75], [[-0.8, 0.8], [-0.8, 0.8]]]
signal = [r'$W^+$',r'$W^-$']
#signal = [r'$W^+$',r'$Z$']
best_model_dir = '/home/samhuang/ML/best_model/'
sample_folder = '/home/samhuang/ML/sample/'
back='_E'
if signal==[r'$W^+$',r'$Z$']:
    save_model_name = best_model_dir+'best_model_binary-WpZ_CNNsq_kappa0.15'+back+'/'
    data_folder = sample_folder+"samples_kappa0.15"+back+"/VBF_H5pp_ww_jjjj_and_VBF_H5z_zz_jjjj/"
elif signal==[r'$W^+$',r'$W^-$']:
    save_model_name = best_model_dir+'best_model_binary-WpWm_CNNsq_kappa0.15'+back+'/'
    data_folder = sample_folder+"samples_kappa0.15"+back+"/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj/"
# Input datasets
#sample_folder = '/home/samhuang/ML/sample/'
#data_folder = sample_folder+"samples_kappa0.15/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj/"
#data_folder = sample_folder+"samples_kappa0.15/VBF_H5pp_ww_jjjj_and_VBF_H5z_zz_jjjj/"

data_tr = data_folder+"train.tfrecord"
data_vl = data_folder+"valid.tfrecord" 
data_te = data_folder+"test.tfrecord" 


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

print ("")
print ("")
print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print ("New Run")
e = datetime.now()
print ("Current date and time = %s" % e)
print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#######################  main code  #############################

# Datasets
dataset_tr, tr_total = get_dataset(data_tr, repeat=False, 
                                   batch_size=batch_size, 
                                   dim_image=dim_image+[True], 
                                   shuffle=shuffle_size_tr, N_labels=N_labels)
dataset_vl, vl_total = get_dataset(data_vl, repeat=False, 
                                   batch_size=batch_size, 
                                   dim_image=dim_image+[True], 
                                   shuffle=0, N_labels=N_labels)
'''
data_id=0
for dd in dataset_tr:
    print (dd[0].shape)
    print (dd[1].shape)
    break
    for i in range(np.shape(dd[0])[0]):
        #print (dd[0][0].shape)
        datanp1 = np.array(dd[0][0][:,:,0])
        datanp2 = np.array(dd[0][0][:,:,0])
        #print (datanp)
        #print (datanp[datanp!=0])
        if (len(datanp1[datanp1!=0])==0) or (len(datanp2[datanp2!=0])==0):
            print (data_id, i)
        data_id += 1
    #print (dd[0].shape)
'''

# Create the model  
model = models.CNNsq(dim_image=dim_image[0] + [2], n_class=2)
#print(model.summary())
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0002, verbose=1, patience=patience)
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
check_point    = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)


history = model.fit(dataset_tr, validation_data=dataset_vl , epochs=train_epochs, batch_size=batch_size, callbacks=[early_stopping, check_point])
print_layer_and_params(model, history)
#model.save(save_model_name)


os.system('mkdir '+save_model_name+'/figures')
# Plot results curves.
fig = plt.figure(1, figsize=(10, 14))
fig.clf()
x = range(len(history.history['loss']))
# Loss.
y_tr = history.history['loss']
y_vl = history.history['val_loss']
ax = fig.add_subplot(2, 1, 1)
ax.plot(x, y_tr, "b", label="Training")
ax.plot(x, y_vl, "b--", label="Validation")
ax.set_title("Loss across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Loss (categorical cross-entropy)")
ax.legend()
# Accuracy.
y_tr = history.history['accuracy']
y_vl = history.history['val_accuracy']
ax = fig.add_subplot(2, 1, 2)
ax.plot(x, y_tr, "b", label="Training")
ax.plot(x, y_vl, "b--", label="Test")
ax.set_title("Accuracy across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Accuracy")
fig.savefig(save_model_name+'/figures/loss and accuracy.png', dpi=300)
plt.close()


loaded_model = tf.keras.models.load_model(save_model_name)

dataset_te, te_total  = get_dataset(data_te, repeat=False, 
                                    batch_size=1, 
                                    dim_image=dim_image+[True], 
                                    shuffle=0, N_labels=N_labels)

results = loaded_model.evaluate(dataset_te)
print("Testing Loss = {0:f}, Testing Accuracy = {1:f}".format(results[0], results[1]))

labels = [x[1][0].tolist() for x in dataset_te.as_numpy_iterator()]

dataset_te, te_total  = get_dataset(data_te, repeat=False, 
                                    batch_size=1, 
                                    dim_image=dim_image+[False], 
                                    shuffle=0, N_labels=N_labels)

predictions = loaded_model.predict(dataset_te).tolist()

data = {'test_scores': predictions, 'test_labels': labels}

print (predictions[:10])
print (labels[:10])



labels = np.array(labels)
predictions = np.array(predictions)
n_class = np.shape(labels)[1]
print (n_class)
    
 
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
 
fig = plt.figure(figsize=(8,6))
for i in range(n_class):
    print ('{0} (auc = {1:0.4f})'.format(signal[i], roc_auc[i]))
    plt.plot(fpr[i], tpr[i], label='{0} (auc = {1:0.2f})'.format(signal[i], roc_auc[i]))
            
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right', fontsize=15)
fig.savefig(save_model_name+'/figures/roc_auc.png', dpi=300)







##############################################################
sys.stdout.close()

os.system("./../best_model/organize_model_log.sh "+save_model_name+'latest_run.log')
os.system("cat "+save_model_name+"latest_run.log >> "+save_model_name+data_folder.split('/')[5]+'.log')

