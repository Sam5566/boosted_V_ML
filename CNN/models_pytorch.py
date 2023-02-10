#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from re import X

import numpy as np
import torch as th
#from torchvision import transforms

class CNN_torch(th.nn.Module):
    def __init__(self, name="CNN_torch", dim_image=(None, 2, 75, 75), n_class=3):
        super(CNN_torch, self).__init__()
        """h2ptjl Channel"""
        self.h2ptjl = th.nn.Sequential(
            #transforms.Lambda(lambda x: x[:, :, :, :]),
            th.nn.BatchNorm2d(2),
            th.nn.Conv2d(dim_image[1], 32, 6, padding='same'),
            th.nn.ReLU(),
            th.nn.MaxPool2d(2,2),
            th.nn.Conv2d(32, 128, 4, padding='same'),
            th.nn.ReLU(),
            th.nn.MaxPool2d(2,2),
            th.nn.Conv2d(128, 256, 6, padding='same'),
            th.nn.ReLU(),
            th.nn.MaxPool2d(2,2),
            th.nn.Dropout(0.1),
        )
        self.linear1 = th.nn.Linear(256*9*9, 512)
        self.linear2 = th.nn.Linear(512, 32)
        self.linear3 = th.nn.Linear(512, 128)
        self.linear4 = th.nn.Linear(128, 32)
        """Output Layer"""
        self._output = th.nn.Linear(32, n_class)
        self.kappa0 = th.nn.Parameter(th.tensor([0.]))
        self.finalized_kappa = th.tensor([-1])

        self.do_dynamic_kappa = True
        self.make_Qk_image = True
        self.dim_image = dim_image

    
    def kappa_transformation(self, kappa0):
        x = th.nn.Sigmoid()(kappa0)
        a = 100
        b = 0#-100
        c = 0.2-25 +0.85
        return  a*x**2 + b*x + c

    def making_plot(self, x, x2, pT_jet, kappa):
        hQk = th.sum(x2 * (pT_jet.reshape((len(pT_jet),200,1,1)))**kappa, dim=1)   
        #print (hQk.size())
        x[:,1] = hQk
        return x

    def forward(self, x, x2, pT_jet): # x: (pT, Qk), x2: Q
        """h2ptjl"""
        #print ("check0")
        #latent_h2ptjl = self.h2ptjl(inputs)
        if self.make_Qk_image == False:
            pass
        elif self.training and self.do_dynamic_kappa: #this bool variable is set in the default model __init__
            kappa = self.kappa_transformation(self.kappa0)
            x = self.making_plot(x, x2, pT_jet, kappa)
        elif (self.do_dynamic_kappa):
            kappa = self.finalized_kappa
            x = self.making_plot(x, x2, pT_jet, kappa)
        
        x = self.h2ptjl(x)  
        #print (x.size())
        x = th.flatten(x,1)
        #print (x.size())
        x = self.linear1(x)
        x = th.nn.ReLU()(x)
        x = th.nn.Dropout(0.5)(x)
        #print (x.size())
        x = self.linear2(x)
        x = th.nn.ReLU()(x)
        x = th.nn.Dropout(0.5)(x)

        # ADDITIONAL 
        #x = self.linear3(x)
        #x = th.nn.ReLU()(x)
        #x = th.nn.Dropout(0.3)(x)

        #x = self.linear4(x)
        #x = th.nn.ReLU()(x)
        #x = th.nn.Dropout(0.3)(x)

        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        latent_all = x
        
        #return self._output(latent_all)
        
        return th.nn.Softmax(dim=1)(self._output(latent_all))

class ResidualBlock(th.nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.left = th.nn.Sequential(
            th.nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding),
            #th.nn.BatchNorm2d(outchannel),
            th.nn.ReLU(inplace=True),
            th.nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding),
            #th.nn.BatchNorm2d(outchannel)
        )
        self.shortcut = th.nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = th.nn.Sequential(
                th.nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, padding=padding),
                #th.nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = th.nn.functional.relu(out)
        return out

class CNN2_torch(th.nn.Module):
    def __init__(self, name="CNN2_torch", dim_image=(None, 2, 75, 75), n_class=3):
        super(CNN2_torch, self).__init__()
        self.inchannel = 32
        """h2ptjl Channel"""
        self.h2ptjl = th.nn.Sequential(
            #transforms.Lambda(lambda x: x[:, :, :, :]),
            th.nn.BatchNorm2d(2),
            th.nn.Conv2d(dim_image[1], 32, 6, padding='same'),
            th.nn.ReLU(),
            th.nn.MaxPool2d(2,2),
            #th.nn.Conv2d(32, 128, 4, padding='same'),
            #th.nn.ReLU(),
            self.make_layer(ResidualBlock, 128,  2, kernal_size=6, stride=1, padding='same'),
            th.nn.MaxPool2d(2,2),
            #th.nn.Conv2d(128, 256, 6, padding='same'),
            #th.nn.ReLU(),
            self.make_layer(ResidualBlock, 256,  2, kernal_size=6, stride=1, padding='same'),
            th.nn.MaxPool2d(2,2),
            th.nn.Dropout(0.1),
        )
        
        self.linear1 = th.nn.Linear(256*9*9, 512)
        self.linear2 = th.nn.Linear(512, 32)
        self.linear3 = th.nn.Linear(512, 128)
        self.linear4 = th.nn.Linear(128, 32)

        """Output Layer"""
        self._output = th.nn.Linear(32, n_class)
        self.kappa0 = th.nn.Parameter(th.tensor([0.]))
        self.finalized_kappa = th.tensor([-1])

        self.do_dynamic_kappa = True
        self.make_Qk_image = True
        self.dim_image = dim_image

    def make_layer(self, block, channels, num_blocks, kernal_size, stride, padding):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, kernal_size, stride, padding))
            self.inchannel = channels
        return th.nn.Sequential(*layers)
    
    def kappa_transformation(self, kappa0):
        x = th.nn.Sigmoid()(kappa0)
        a = 100
        b = 0#-100
        c = 0.2-25 +0.85
        return  a*x**2 + b*x + c

    def making_plot(self, x, x2, pT_jet, kappa):
        hQk = th.sum(x2 * (pT_jet.reshape((len(pT_jet),200,1,1)))**kappa, dim=1)   
        #print (hQk.size())
        x[:,1] = hQk
        return x


    def forward(self, x, x2, pT_jet):
        """h2ptjl"""
        #print ("check0")
        #latent_h2ptjl = self.h2ptjl(inputs)
        
        if self.make_Qk_image == False:
            pass
        elif self.training and self.do_dynamic_kappa: #this bool variable is set in the default model __init__
            kappa = self.kappa_transformation(self.kappa0)
            x = self.making_plot(x, x2, pT_jet, kappa)
        elif (self.do_dynamic_kappa):
            kappa = self.finalized_kappa
            x = self.making_plot(x, x2, pT_jet, kappa)
        
        x = self.h2ptjl(x)  
        #print (x.size())
        x = th.flatten(x,1)
        #print (x.size())
        x = self.linear1(x)
        x = th.nn.ReLU()(x)
        x = th.nn.Dropout(0.5)(x)
        #print (x.size())
        x = self.linear2(x)
        x = th.nn.ReLU()(x)
        x = th.nn.Dropout(0.5)(x)

        
        """Output"""
        latent_all = x
        
        
        return th.nn.Softmax(dim=1)(self._output(latent_all))
