import numpy as np
import json

class card:
    def __init__(self,save_model_name):
        self.save_model_name = save_model_name
        
        # the type of parameters that are going to be save
        self.archive = {}
        self.archive['input information'] = {}
        self.archive['hyperparameters'] = {}
        self.archive['history'] = {}
        self.archive['AUC'] = {}
        self.archive['ACC'] = {}
        self.archive['testing result'] = {}
        self.archive['best model'] = {}

    def save_into_card(self, dict_name=np.nan, **arg):
        #print (arg)
        if dict_name is np.nan:
            print ("Error: The dict_name should be indicated!")
            print ("The currently valid dict_name are", self.archive.keys())
            print ("If you want to add a new branch, just type the name to dict_name, and the card class will archive it.")
            return
        elif dict_name in self.archive.keys():
            self.archive[dict_name].update(arg)
        else:
            print ("Warning: Saving a new branch to the class")
            self.archive[dict_name] = {}
            self.archive[dict_name].update(arg)
        #print (self.archive)

    def output_card(self):
        with open(self.save_model_name+'training_information.json', 'w') as fp:
            json.dump(self.archive, fp)

    def read_card(self, read_model_name):
        print ("Will read the training information from the existing card ({:s}), and the current card variable will be overwrited!")
        with open(read_model_name+'training_information.json', 'r') as fp:
                self.archive = json.load(fp)
        print (self.archive)

