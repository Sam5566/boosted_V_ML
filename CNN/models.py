from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class CNN_ternary(tf.keras.Model):
    def __init__(self, name="CNN_ternary", dim_image=(75, 75, 2)):
        super(CNN_ternary, self).__init__(name=name)
        
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
        self._output = tf.keras.layers.Dense(3, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptjl"""
        latent_h2ptjl = self.h2ptjl(inputs)
        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        latent_all = latent_h2ptjl
        
        return self._output(latent_all)



class CNN_binary(tf.keras.Model):
    def __init__(self, name="CNN_binary", dim_image=(75, 75, 2)):
        super(CNN_binary, self).__init__(name=name)
        
        """h2ptjl Channel"""
        self.h2ptjl = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, :, :, :2]),
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
        self._output = tf.keras.layers.Dense(2, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptjl"""
        latent_h2ptjl = self.h2ptjl(inputs)
        
        """Output"""
        #latent_all = tf.concat([latent_h2ptj, latent_h2ptjl], axis=1)
        latent_all = latent_h2ptjl
        
        return self._output(latent_all)


class CNNsq_ternary(tf.keras.Model):
    def __init__(self, name="CNNsq_ternary", dim_image=(75, 75, 2)):
        super(CNNsq_ternary, self).__init__(name=name)
        
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
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
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
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(3, activation='softmax')
        
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


class CNNsq_binary(tf.keras.Model):
    def __init__(self, name="CNNsq_binary", dim_image=(75, 75, 2)):
        super(CNNsq_binary, self).__init__(name=name)
        
        """h2ptj Channel"""
        self.h2ptj = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, :, :, 0]),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :,0], -1)),
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
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
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
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(2, activation='softmax')
        
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
