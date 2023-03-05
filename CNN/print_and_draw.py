import keras.backend as K
from cprint import *
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')
import pandas as pd
import numpy as np
import tensorflow as tf
import torch as th

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix,accuracy_score, classification_report
from scipy import interpolate
import sys
sys.path.insert(0, '/home/samhuang/ML/analysis')
from SmoothGradCAMplusplus.cam import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp
from SmoothGradCAMplusplus.utils.visualize import visualize, reverse_normalize
from SmoothGradCAMplusplus.utils.imagenet_labels import label2idx, idx2label
from torchvision import transforms

import pdb


################################
def print_layer_and_params(model, history):
    moneyline = "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print (moneyline+"\n"+moneyline)
    print(model.summary(print_fn=cprint.info))

    for id_layer, layer in enumerate(model.layers):
        try:
        #if (1):
            print ("\n@LAYER"+str(id_layer+1)+"       @@@@@@@@@@@@@@@@@@@@@@")
            print (layer.summary(print_fn=cprint.ok))
            cprint.info ("%Optimizer:\n", model.optimizer.get_config())
            cprint.info ("%Layer detail:\n", layer.get_config())
        
        except:
            cprint.info ("%Layer detail:\n", layer.get_config())

    print ("\n****************************************************")
    print ("history keys:\n", history.history.keys())
    print ("history params:\n", history.params)
    print ("****************************************************")
   
    print (moneyline+"\n"+moneyline)


def orgainize_result_table(model_directory_name):
    mdn = model_directory_name
    os.system(" echo ")


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        if self.terminal != None:
            sys.stdout = self.terminal
            self.terminal = None
        if self.log != None:
            self.log.close()
            self.log = None
    def give_model_log_directory(self, save_model_name):
        os.system('mkdir '+save_model_name)
        self.log = open(save_model_name+'latest_run.log', "w+")

def plot_training_history(collection_history, fig):
    df = pd.DataFrame()
    datanp = []
    xmax = 0
    for id_history, history in enumerate(collection_history):
        x = range(len(history.history['loss']))
        xmax = max(xmax, len(history.history['loss']))
        y_tr = history.history['loss']
        y_vl = history.history['val_loss']
        yacc_tr = history.history['accuracy']
        yacc_vl = history.history['val_accuracy']
        datanp.append([x, y_tr, y_vl, yacc_tr, yacc_vl])

    epoch_information = []
    means = [[],[],[],[]]
    errors = [[],[],[],[]]
    x = np.array(range(xmax))+1
    for epoch in range(xmax):
        epoch_information.append([[], [], [], []])
        for id_history, history in enumerate(collection_history):
            if len(datanp[id_history][0]) <= epoch:
                continue
            for ii in range(len(datanp[id_history])-1):
                #print ("ii =", ii, "epoch =", epoch, "id_history =", id_history)
                #print (len(datanp[id_history][ii+1]))
                #print (datanp[id_history][ii+1][epoch])
                epoch_information[-1][ii].append(datanp[id_history][ii+1][epoch])
        for ii in range(len(datanp[id_history])-1):
            means[ii].append(np.mean(epoch_information[-1][ii]))
            errors[ii].append(np.std(epoch_information[-1][ii]))
    means = np.array(means)
    errors = np.array(errors)
            
    # plotting
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(x, means[0], yerr=errors[0], label="Training")
    ax.errorbar(x, means[1], yerr=errors[1], label="Validation")
    ax.axvline(x=x[-10], color="black", linestyle="--")
    ax.set_title("Loss across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (categorical cross-entropy)")
    ax.legend()

    ax = fig.add_subplot(2, 1, 2)
    ax.errorbar(x, means[2]*100, yerr=errors[2]*100, label="Training")
    ax.errorbar(x, means[3]*100, yerr=errors[3]*100, label="Validation")
    ax.set_title("Accuracy across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()

    return [means, errors], fig

def calculate_ACC(collection_predictions, collection_labels, classes):
    n_class = np.shape(collection_labels[0])[1]
    print ("N of classes", n_class)
    if (n_class != len(classes)):
        print ("Error of signal classes")

    
    #accs = []
    performance_report = []
    accs = [[] for i in range(n_class)]
    for j in range(len(collection_labels)):
        performance_report.append(classification_report(np.argmax(collection_labels[j], axis=1), np.argmax(collection_predictions[j], axis=1), target_names=classes, output_dict=True, zero_division=0))
        
        for i_class in range(n_class):
            accs[i_class].append(performance_report[-1][classes[i_class]]['precision'])
    #       print (np.argmax(collection_labels[j][:, i_class]))
    #        acc[i_class] = accuracy_score(np.argmax(collection_labels[j][:, i_class]), np.argmax(collection_predictions[j][:, i_class]))

    return accs

def plot_roc_curve(collection_predictions, collection_labels, classes, fig):
    n_class = np.shape(collection_labels[0])[1]
    print ("N of classes", n_class)
    if (n_class != len(classes)):
        print ("Error of signal classes")

    fprs, tprs, roc_auc_values = [], [], []
    for j in range(len(collection_labels)):
        fpr, tpr, roc_auc = [[] for i in range(n_class)], [[] for i in range(n_class)], [[] for i in range(n_class)]  
        for i_class in range(n_class):
            fpr[i_class], tpr[i_class], _ = roc_curve(collection_labels[j][:, i_class], collection_predictions[j][:, i_class])
            roc_auc[i_class] = auc(fpr[i_class], tpr[i_class])

            f = interpolate.interp1d(fpr[i_class], tpr[i_class])
            xnew = np.arange(0, 1, 0.01)
            ynew = f(xnew)
            fpr[i_class] = xnew.tolist()
            tpr[i_class] = ynew.tolist()
        fprs.append(fpr)
        tprs.append(tpr)
        roc_auc_values.append(roc_auc)
    
    roc_auc_values = np.array(roc_auc_values)
    fprs = np.array(fprs)
    tprs = np.array(tprs)

    for i in range(n_class):
        print ('{0} (auc = {1:2.2f} +- {2:.4f} %)'.format(classes[i], np.mean(roc_auc_values[:,i])*100, np.std(roc_auc_values[:,i]*100)))
        plt.errorbar(np.mean(fprs[:,i], axis=0), np.mean(tprs[:,i], axis=0), yerr=np.std(tprs[:,i], axis=0), label=r'{0} (auc = {1:2.2f}$\pm${2:.4f} %)'.format(classes[i], np.mean(roc_auc_values[:,i])*100, np.std(roc_auc_values[:,i]*100)))

    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right', fontsize=15)
    return roc_auc_values, fig

def plot_confusion_matrix(collection_predictions, collection_labels, classes, ax):
    #Confusion matrix
    collection_cm = []
    for j in range(len(collection_labels)):
        pred = tf.argmax(collection_predictions[j], axis = 1)
        label = tf.argmax(collection_labels[j], axis = 1)
        cm=confusion_matrix(pred,label)
        collection_cm.append(cm)
    
    collection_cm = np.array(collection_cm)
    disp=ConfusionMatrixDisplay(confusion_matrix=np.mean(collection_cm, axis=0),display_labels=classes)

    disp.plot(ax=ax)
    return collection_cm

def plot_SmoothGrad(model, input_imag_loader, collection_predictions, collection_labels, classes, n_imag=1):
    normalize = transforms.Normalize(
                mean=[0.485], std=[0.229]
                #mean=[0.485, 0.485],
                #std=[0.229, 0.229]
                ) # N channel

    preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize
                ])
    
    for i, (images, images2, pTnorms, labels) in enumerate(input_imag_loader):
        print (images.shape)
        print (th.min(images2), th.max(images2))
        print (images)
        for j, imag in enumerate(images):
            tensor = preprocess(imag[0].numpy())
            print (tensor.shape)

            # reshape 4D tensor (N, C, H, W)
            tensor = tensor.unsqueeze(0)
            print (tensor.shape)

        model.eval()
        print(model)
        print (dir(model))

        # the target layer you want to visualize
        target_layer = model.h2ptjl
        print (target_layer[-1])

        # wrapper for class activation mapping. Choose one of the following.
        # wrapped_model = CAM(model, target_layer)
        # wrapped_model =GradCAM(model, target_layer)
        # wrapped_model = GradCAMpp(model, target_layer)
        wrapped_model = SmoothGradCAMpp(model, target_layer, n_samples=1, stdev_spread=0.15)

        cam, idx = wrapped_model(tensor, images2, pTnorms)
        print(idx2label[idx])

        if (i >= n_imag):
            break
    return 0


def draw_resonance_peak(model, test_labels, test_predictions, signal, extra_inputs, extra_information, fig):
    Nbin = 25
    df = pd.DataFrame()
    df['labels'] = np.argmax(test_labels, axis=1)
    df['predictions'] = np.argmax(test_predictions, axis=1)
    for iii in range(len(extra_information)):
        df[extra_information[iii]] = extra_inputs[:,iii]
        max_input, min_input = max(df[extra_information[iii]]), min(df[extra_information[iii]])
        bins_loc = np.linspace(min_input, max_input, Nbin+1)
        dflist_labels = []
        dflist_preds = []
        for ii in range(len(test_labels[0])):
            dflist_preds.append(df[df['predictions']==ii][extra_information[iii]])
            dflist_labels.append(df[df['labels']==ii][extra_information[iii]])

        # plotting
        

        ax = fig.add_subplot(2, len(extra_information), 1+2*iii)
        n0, _, _ = ax.hist(dflist_labels[:-1], Nbin, stacked=True, range=[min_input, max_input])
        ax.cla()
        #print (n0)
        n1, _, _ = ax.hist(dflist_preds, Nbin, stacked=True, range=[min_input, max_input])
        #print (n1)
        plt.plot(bins_loc[:-1]+0.5*(bins_loc[1]-bins_loc[0]), n0[-1], 'ko')
        plt.plot(bins_loc[:-1]+0.5*(bins_loc[1]-bins_loc[0]), n0[-1], 'k_', markersize=15, label='_nolegend_')
        ax.set_xlim(min_input*0.7, max_input*1.1)
        #ax.set_ylim(1e-1, np.max(n1))
        ax.set_title("resonance peak")
        #ax.set_yscale('log')
        ax.set_ylabel(r"$N/$"+"{:.2f}".format(bins_loc[1]-bins_loc[0])+" (GeV$^{-1}$)")
        ax.get_xaxis().set_visible(False)
        ax.legend(['simulation']+signal)

        gs = gridspec.GridSpec(5,len(extra_information))
        ax.set_position(gs[0:3].get_position(fig))
        ax.set_subplotspec(gs[0:3])              # only necessary if using tight_layout()

        #ax = fig.add_subplot(2, len(extra_information), 2+2*iii)
        ax = fig.add_subplot(gs[3])
        ax.axhline(y=1, color="black", linestyle="-")
        ax.axhline(y=0.9, color="black", linestyle="--")
        ax.axhline(y=1.1, color="black", linestyle="--")
        ax.plot(bins_loc[:-1]+0.5*(bins_loc[1]-bins_loc[0]), n1[-2]/n0[-1], 'ko')
        ax.set_xlim(min_input*0.7, max_input*1.1)
        ax.set_ylim(0.8, 1.2)
        ax.set_ylabel(r"$N_\mathrm{obs}/N_\mathrm{sim}$")
        ax.set_xlabel(extra_information[iii])
        ax.get_xaxis().set_visible(False)

        ax = fig.add_subplot(gs[4])
        ax.axhline(y=1, color="black", linestyle="--")
        for ii in range(len(test_labels[0])-1):
            if ii>0:
                ax.plot(bins_loc[:-1]+0.5*(bins_loc[1]-bins_loc[0]), (n1[ii]-n1[ii-1])/(n0[ii]-n0[ii-1]), '_', markersize=15)
            else:
                ax.plot(bins_loc[:-1]+0.5*(bins_loc[1]-bins_loc[0]), n1[ii]/n0[ii], '_', markersize=15)
        ax.set_xlim(min_input*0.7, max_input*1.1)
        #ax.set_ylim(0.8, 1.2)
        ax.set_ylabel(r"$N^\mathrm{class}_\mathrm{obs}/N_\mathrm{sim}$")
        ax.set_xlabel(extra_information[iii]+"(GeV)")
        #ax.legend(['simulation']+signal)

    return fig
