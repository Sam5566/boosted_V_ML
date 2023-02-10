from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import GPU_selection
import os
os.environ['CUDA_VISIBLE_DEVICES']=str(GPU_selection.pick_gpu_lowest_memory())#'2'
import sys
os.chdir('/home/samhuang/ML/CNN/')
sys.path.insert(0, '/home/samhuang/ML')
sys.path.insert(0, '/home/samhuang/ML/sample')
#sys.path.insert(0, '/home/samhuang/../public/Polar_new/samples')
from readTFR import *
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc
import models
import numpy as np
import tensorflow as tf
import json
from train_utils import *
from print_and_draw import *
from datetime import datetime, date
import pdb

physical_gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_gpus))
tf.config.experimental.set_memory_growth(physical_gpus[0], True)
#################  input and variables  #######################
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
best_model_dir = '/home/samhuang/ML/best_model/'
save_model_name = best_model_dir+'best_model_ternary_CNN_kappa0.15/'
signal=[r'$W^+$',r'$W^-$',r'$Z$']
N_signal = len(signal)

# Input datasets
sample_folder = '/home/samhuang/ML/sample/jet_base/'
#sample_folder = '/home/samhuang/Fishbone/'
#data_folder = "sample/samples_kappa0.15/samples_kappa0.15/"
data_folder = sample_folder+"samples_kappa0.15/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj/"
#data_folder = "sample/samples_kappa0.15/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj/"
#data_folder = "/home/samhuang/../public/Polar_new/samples/"
#data_folder = "samples/"
data_tr = data_folder+"train.tfrecord"
data_vl = data_folder+"valid.tfrecord" 
data_te = data_folder+"test.tfrecord" 

#data_tr = "/home/public/train.tfrecord"
#data_vl = "/home/public/valid.tfrecord" 
#data_te = "/home/public/test.tfrecord" 

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



collection_history = []
collection_predictions = []
collection_labels = []
collection_test = []
for i_try in range(Ntry):
    # Datasets
    dataset_tr, tr_total = get_dataset(data_tr, repeat=False, 
                                   batch_size=batch_size, 
                                   dim_image=dim_image+[True], 
                                   shuffle=shuffle_size_tr, N_labels=N_signal)
    dataset_vl, vl_total = get_dataset(data_vl, repeat=False, 
                                   batch_size=batch_size, 
                                   dim_image=dim_image+[True], 
                                   shuffle=0, N_labels=N_signal)
    print ("Number of training datasets: %d." %tr_total)

    model = models.CNN3(dim_image=dim_image[0] + [2], n_class=N_signal)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0002, verbose=1, patience=patience)
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
    check_point    = tf.keras.callbacks.ModelCheckpoint(save_model_name+'Try/'+str(i_try), monitor='val_loss', verbose=1, save_best_only=True)


    history = model.fit(dataset_tr, validation_data=dataset_vl , epochs=train_epochs, batch_size=batch_size, callbacks=[early_stopping, check_point])

    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = history.history
    # Save it under the form of a json file
    json.dump(history_dict, open(save_model_name+'Try/'+str(i_try)+'/history.json', 'w'))

    print_layer_and_params(model, history)
    #pdb.set_trace()
    tf.keras.utils.plot_model(model, to_file=save_model_name+'figures/model.png', show_shapes=True, expand_nested=True, show_layer_names=True, dpi=300)
    collection_history.append(history)

    loaded_model = tf.keras.models.load_model(save_model_name+'Try/'+str(i_try))
    dataset_te, te_total  = get_dataset(data_te, repeat=False, 
                                    batch_size=1, 
                                    dim_image=dim_image+[True], 
                                    shuffle=0, N_labels=N_signal)

    results = loaded_model.evaluate(dataset_te)
    print("Testing Loss = {0:f}, Testing Accuracy = {1:f}".format(results[0], results[1]))
    collection_test.append(results)

    labels = [x[1][0].tolist() for x in dataset_te.as_numpy_iterator()]
    collection_labels.append(np.array(labels))

    dataset_te, te_total  = get_dataset(data_te, repeat=False, 
                                    batch_size=1, 
                                    dim_image=dim_image+[False], 
                                    shuffle=0, N_labels=N_signal)

    predictions = loaded_model.predict(dataset_te).tolist()
    collection_predictions.append(np.array(predictions))
    #data = {'test_scores': predictions, 'test_labels': labels}
    



os.system('mkdir '+save_model_name+'figures')
# Plot results curves.
fig = plt.figure(1, figsize=(10, 14))
fig.clf()
fig = plot_training_history(collection_history, fig)

fig.savefig(save_model_name+'/figures/loss and accuracy.png', dpi=300)
plt.close()

fig = plt.figure(figsize=(8,6))
roc_auc_values, fig = plot_roc_curve(collection_predictions, collection_labels, signal, fig)
fig.savefig(save_model_name+'/figures/roc_auc.png', dpi=300)

collection_accs = calculate_ACC(collection_predictions, collection_labels, signal)
for i_class, classi in enumerate(collection_accs):
    print (signal[i_class], "(acc = {:.2f} +- {:.4f} %".format(np.mean(classi)*100, np.std(classi)*100))

collection_test = np.array(collection_test)
print ("The summarized testing accuracy = {:.2f} +- {:.4f} %, with the loss = {:.4f} +- {:.6f}".format(np.mean(collection_test[:,1]*100), np.std(collection_test[:,1]*100), np.mean(collection_test[:,0]), np.std(collection_test[:,0])))





##############################################################
sys.stdout.close()

os.system("./../best_model/organize_model_log.sh "+save_model_name+'latest_run.log')
os.system("cat "+save_model_name+"latest_run.log >> "+save_model_name+data_folder.split('/')[5]+'.log')
#os.system('ls -lh '+save_model_name+'*.log')
