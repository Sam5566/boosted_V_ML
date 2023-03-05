#!/usr/bin/env python
# coding: utf-8
# python print_model_layer <resolution> <rescalse>
# python print_model_layer 15 False

# sample path & model path should be changed

import os
import sys
sys.path.insert(0, '/home/samhuang/ML/CNN')
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras.utils import image

from matplotlib import colors
from readTFR import *

def main():
    kappa = 0.15
    nevent = '1000k'
    # resolution of image
    res = int(sys.argv[1])
    rescale = True if sys.argv[2]=='True' else False
    
    batch_size = 512
    if rescale:
        dim_image = [[75, 75], [[-3, 3], [-3, 3]]]
        res = f'{res}x{res}-75x75'
    else:
        dim_image = [[res, res], [[-3, 3], [-3, 3]]]
        res = f'{res}x{res}'
    
    # input sample path
    sample_dir = f'/home/r10222035/Boosted_V/sample/event_samples_kappa{kappa}-{nevent}-{res}/'
    sample_dir = '../sample/event_base/samples_kappa0.15/VBF_H5pp_ww_jjjj_and_VBF_H5mm_ww_jjjj_and_VBF_H5z_zz_jjjj_and_VBF_H5z_ww_jjjj_and_VBF_H5p_wz_jjjj_and_VBF_H5m_wz_jjjj/'
    print(f'Read data from {sample_dir}')
    data_te = os.path.join(sample_dir, 'test.tfrecord')
    dataset_te, te_total = get_dataset(data_te, repeat=False, 
                                       batch_size=batch_size, 
                                       dim_image=dim_image+[True], 
                                       shuffle=0,
                                       N_labels=6,
                                      )
    
    for x in dataset_te.take(10):
        img = x[0][0]
    x = np.array(img)
    x = np.expand_dims(x, axis=0)

    # Model path
    save_model_name = f'best_model/best_model_event_CNN_kappa{kappa}-{nevent}-{res}/'
    save_model_name = '/home/samhuang/ML/best_model/best_model_ternary_CNN_event_kappa0.15/Try/0'
    print(f'Load model from {sample_dir}')
    CNN = tf.keras.models.load_model(save_model_name)
    
    layer_outputs = []
    for layer in CNN.layers:
        try:
            for layer2 in layer.output:
            	layer_outputs.append(layer2.output)
        except:
            pass

    #layer_outputs = [layer.output for layer in CNN.layers]
    layer_names = []
    for layer in CNN.layers:
        layer_names.append(layer.name)
    print(layer_names)
    
    # predict results
    model = tf.keras.models.Model(inputs=CNN.input, outputs=layer_outputs)
    activations = model.predict(x)
    
    # plot middle layer
    n_layer = 7
    fig, ax = plt.subplots(n_layer,1, figsize=(5, n_layer*5))
    for i, (activation,layer_name) in enumerate(zip(activations[0:n_layer],layer_names)):
        h = activation.shape[1]
        w = activation.shape[2]
        num_channels = activation.shape[3]
        c = 0
        while (activation[0,0:h,0:w,c] == 0).all() and c < num_channels-1:
            c += 1
        ax[i].set_title(layer_name,fontsize=16)
        ax[i].imshow(activation[0,0:h,0:w,c], origin='lower', norm=colors.SymLogNorm(linthresh=0.1, linscale=0.6, vmin=0), cmap='Blues')
        ax[i].xaxis.set_major_locator(plt.MaxNLocator(8))
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(8))

    plt.savefig(f'figures/CNN_middle_layer_output_{res}.png', facecolor='White', dpi=300, bbox_inches = 'tight')

if __name__ == '__main__':
    main()
