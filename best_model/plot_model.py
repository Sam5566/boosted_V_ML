import pandas as pd
import tensorflow as tf
from tensorflow import keras

###################################################
save_model_name = 'best_model_ternary_CNN_kappa0.15_2jet'

###################################################
#def layer_read(layer):
#    try:
#        for sublayer in layer.layers:

###################################################
loaded_model = tf.keras.models.load_model(save_model_name)

all_layers = []
print ("layer N:",len(loaded_model.layers))
for layer in loaded_model.layers:
    print ()
    try:
        all_layers.extend(layer.layers)
    except:
        all_layers.append(layer)
print (all_layers)
gan_plot = tf.keras.models.Sequential(all_layers)
exit()
print ("***********************************")
gan_plot.build(((None,75,75,4)))
print (gan_plot.summary())
keras.utils.plot_model(gan_plot, to_file=save_model_name+'/figures/model.png', show_shapes=True,expand_nested=True, show_layer_names=True, dpi=300)

