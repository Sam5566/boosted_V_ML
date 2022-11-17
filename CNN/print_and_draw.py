import keras.backend as K
from cprint import *
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from scipy import interpolate


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
        
            
    # plotting
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(x, means[0], yerr=errors[0], label="Training")
    ax.errorbar(x, means[1], yerr=errors[1], label="Validation")
    ax.set_title("Loss across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (categorical cross-entropy)")
    ax.legend()

    ax = fig.add_subplot(2, 1, 2)
    ax.errorbar(x, means[2], yerr=errors[2], label="Training")
    ax.errorbar(x, means[3], yerr=errors[3], label="Validation")
    ax.set_title("Accuracy across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Accuracy")
    ax.legend()

    return fig


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