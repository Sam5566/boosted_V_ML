from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import X

import tensorflow as tf
import torch as th
from torchvision import transforms

class CNN(tf.keras.Model):
    def __init__(self, name="CNN", dim_image=(75, 75, 2), n_class=3):
        super(CNN, self).__init__(name=name)
        
        """h2ptjl Channel"""
        self.h2ptjl = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, :, :, :]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (6,6), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(n_class, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptjl"""
        latent_h2ptjl = self.h2ptjl(inputs)
        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        latent_all = latent_h2ptjl
        
        return self._output(latent_all)

class CNN_torch(th.nn.Module):
    def __init__(self, name="CNN_torch", dim_image=( None, 2, 75, 75), n_class=3):
        super(CNN_torch, self).__init__()
        
        """h2ptjl Channel"""
        # self.h2ptjl = th.nn.Sequential(
        #     #transforms.Lambda(lambda x: x[:, :, :, :]),
        #     th.nn.BatchNorm2d(2),
        #     # th.nn.Conv2d(dim_image[-1], 32, 6, padding='same'),
        #     # th.nn.ReLU(),
        #     # th.nn.MaxPool2d(2,2),
        #     # th.nn.Conv2d(32, 128, 4, padding='same'),
        #     # th.nn.ReLU(),
        #     # th.nn.MaxPool2d(2,2),
        #     # th.nn.Conv2d(128, 256, 6, padding='same'),
        #     # th.nn.ReLU(),
        #     # th.nn.MaxPool2d(2,2),
        #     # th.nn.Dropout(0.1),
        #     # th.nn.Flatten(),
        #     #th.nn.Linear(512),
        #     #th.nn.ReLU(),
        #     #th.nn.Dropout(0.5),
        #     #th.nn.Dense(512),
        #     #th.nn.ReLU(),
        #     #th.nn.Dropout(0.5),
        # )
        
        self.ba = th.nn.BatchNorm2d(2)
        """Output Layer"""
        #self._output = th.nn.Linear(n_class)#, activation='softmax')
        
    @tf.function
    def forward(self, inputs):
        """h2ptjl"""
        print ("check0")
        #latent_h2ptjl = self.h2ptjl(inputs)
        latent_h2ptjl = self.ba(inputs)
        # latent_h2ptjl = 1
        # self.ba(inputs)
        print ("check")
        print (th.size())
        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        latent_all = latent_h2ptjl
        
        return self._output(latent_all)


class CNN2(tf.keras.Model):
    def __init__(self, name="CNN2", dim_image=(75, 75, 2), n_class=3):
        super(CNN2, self).__init__(name=name)
        
        """h2ptjl Channel"""
        self.lambda1 = tf.keras.layers.Lambda(lambda x: x[:, :, :, :])
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(32, (6,6), padding='same', activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPool2D((2,2))
        self.conv2 = tf.keras.layers.Conv2D(128, (4,4), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.h2ptjl = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, :, :, :]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (6,6), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(n_class, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptjl"""
        latent_h2ptjl = self.h2ptjl(inputs)
        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        x = latent_h2ptjl
        # x = self.lambda1(inputs)
        # x = self.batchnorm(x)
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool1(x)
        # x = self.conv3(x)
        # x = self.maxpool1(x)
        # x = self.dropout1(x)
        # x = self.flatten(x)
        # x = self.dense1(x)
        # x = self.dropout2(x)
        # x = self.dense1(x)
        # x = self.dropout2(x)

        
        return self._output(x)

class CNNsq(tf.keras.Model):
    def __init__(self, name="CNNsq", dim_image=(75, 75, 2), n_class=2):
        super(CNNsq, self).__init__(name=name)
        
        """h2ptj Channel"""
        self.h2ptj = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 0]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,0]    , -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        """h2Qkj Channel"""
        self.h2Qkj = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 1]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,1], -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(n_class, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptj"""
        latent_h2ptj = self.h2ptj(inputs)

        """h2Qkj"""
        latent_h2Qkj = self.h2Qkj(inputs)
        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        latent_all = tf.concat([latent_h2ptj, latent_h2Qkj], axis=1)
        
        return self._output(latent_all)

class CNNsqtest(tf.keras.Model):
    def __init__(self, name="CNNsqtest", dim_image=(75, 75, 2), n_class=2):
        super(CNNsqtest, self).__init__(name=name)
        
        """h2ptj Channel"""
        self.h2ptj = tf.keras.Sequential(name = '1', layers=[
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 0]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,0]    , -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        """h2Qkj Channel"""
        self.h2Qkj = tf.keras.Sequential(name = '2', layers=[
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 1]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,1], -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(n_class, activation='softmax')

        self.architecture = [['0', '1', '-1'], ['0','2', '-1']]
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptj"""
        latent_h2ptj = self.h2ptj(inputs)

        """h2Qkj"""
        latent_h2Qkj = self.h2Qkj(inputs)
        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        latent_all = tf.concat([latent_h2ptj, latent_h2Qkj], axis=1)
        
        return self._output(latent_all)




## sam's production
#// concept: use two jet image as one input and divide them into different channels.
class CNN_2jet(tf.keras.Model):
    def __init__(self, name="CNN_2jet", dim_image=(75, 75, 4), n_class=6):
        super(CNN_2jet, self).__init__(name=name)
        
        """h2ptjl Channel"""
        self.h2ptjl = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, :, :, 0:2]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (6,6), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        """h2ptj2 Channel"""
        self.h2ptj2 = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, :, :, 2:]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (6,6), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(n_class, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptjl"""
        latent_h2ptj1 = self.h2ptjl(inputs)

        """h2ptj2"""
        latent_h2ptj2 = self.h2ptj2(inputs)
        
        """Output"""
        latent_all = tf.concat([latent_h2ptj1, latent_h2ptj2], axis=1)
        
        return self._output(latent_all)




class CNNsq_2jet(tf.keras.Model):
    def __init__(self, name="CNNsq_2jet", dim_image=(75, 75, 4), n_class=6):
        super(CNNsq_2jet, self).__init__(name=name)
        
        """h2ptj Channel"""
        self.h2ptj1 = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 0]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,0]    , -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        self.h2ptj2 = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 0]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,2]    , -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        """h2Qkj Channel"""
        self.h2Qkj1 = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 1]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,1], -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        self.h2Qkj2 = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 1]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,3], -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(n_class, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptj1"""
        latent_h2ptj1 = self.h2ptj1(inputs)

        """h2Qkj1"""
        latent_h2Qkj1 = self.h2Qkj1(inputs)
        
        """h2ptj2"""
        latent_h2ptj2 = self.h2ptj2(inputs)

        """h2Qkj"""
        latent_h2Qkj2 = self.h2Qkj2(inputs)
        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        latent_all = tf.concat([latent_h2ptj1, latent_h2ptj2, latent_h2Qkj1, latent_h2Qkj2], axis=1)
        
        return self._output(latent_all)


class CNN_4jet(tf.keras.Model):
    def __init__(self, name="CNN_4jet", dim_image=(75, 75, 8), n_class=6):
        super(CNN_4jet, self).__init__(name=name)
        
        """h2ptjl Channel"""
        self.h2ptjl = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, :, :, 0:2]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (6,6), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        """h2ptj2 Channel"""
        self.h2ptj2 = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, :, :, 2:4]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (6,6), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        """h2ptj2 Channel"""
        self.h2ptj3 = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, :, :, 4:6]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (6,6), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        """h2ptj2 Channel"""
        self.h2ptj4 = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, :, :, 6:8]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (6,6), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(128, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(n_class, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptjl"""
        latent_h2ptj1 = self.h2ptjl(inputs)

        """h2ptj2"""
        latent_h2ptj2 = self.h2ptj2(inputs)
        """h2ptjl"""
        latent_h2ptj3 = self.h2ptj3(inputs)

        """h2ptj2"""
        latent_h2ptj4 = self.h2ptj4(inputs)
        
        """Output"""
        latent_all = tf.concat([latent_h2ptj1, latent_h2ptj2, latent_h2ptj3, latent_h2ptj4], axis=1)
        
        return self._output(latent_all)




class CNNsq_4jet(tf.keras.Model):
    def __init__(self, name="CNNsq_2jet", dim_image=(75, 75, 4), n_class=6):
        super(CNNsq_2jet, self).__init__(name=name)
        
        """h2ptj Channel"""
        self.h2ptj1 = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 0]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,0]    , -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        self.h2ptj2 = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 0]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,2]    , -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        """h2Qkj Channel"""
        self.h2Qkj1 = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 1]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,1], -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        self.h2Qkj2 = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 1]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,3], -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(n_class, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptj1"""
        latent_h2ptj1 = self.h2ptj1(inputs)

        """h2Qkj1"""
        latent_h2Qkj1 = self.h2Qkj1(inputs)
        
        """h2ptj2"""
        latent_h2ptj2 = self.h2ptj2(inputs)

        """h2Qkj"""
        latent_h2Qkj2 = self.h2Qkj2(inputs)
        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        latent_all = tf.concat([latent_h2ptj1, latent_h2ptj2, latent_h2Qkj1, latent_h2Qkj2], axis=1)
        
        return self._output(latent_all)
