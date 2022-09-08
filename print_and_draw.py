import keras.backend as K
from cprint import *
import os

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