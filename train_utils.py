from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
sys.path.insert(0, os.path.expanduser('~')+'/GNN/samples/TFR')
from readTFR import *
from matplotlib import pyplot as plt
import models
import numpy as np
import scipy
import tensorflow as tf
import json
import pandas as pd
import random
from tqdm import tqdm

__all__ = ['create_loss','compute_labels','extract_info']

    
def create_loss(targets, outputs):
    """Global-wise categorical cross entropy loss.
    Returns global-wise categorical cross entropy loss.
    Args:
    target: Target `graph.GraphsTuple`.
    output: Output `graph.GraphsTuple`.
    Returns:
    loss: Global-wise categorical cross entropy loss.
    """
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    loss = cce(targets.globals, outputs.globals)
    
    return loss

def compute_labels(targets, outputs):
    """Calculate global classification accuracy.
    Returns the label list of global attribute predction.
    Args:
    target: A `graphs.GraphsTuple` that contains the target graph.
    output: A `graphs.GraphsTuple` that contains the output graph.
    Returns:
    label_list: A `list` of global attribute labels.
    """
    label_list = []

    for target, output in zip(targets.globals, outputs.globals):
        x = tf.math.argmax(target)
        y = tf.math.argmax(tf.nn.softmax(output))
        label_list.append(x == y)

    return label_list

def extract_info(target_gt, output_gt, in_dict, name):
    """Extract a `dict` containing global testing results.
    Returns a `dict` containing global testing results.
    Args:
    target_op: A `graphs.GraphsTuple` that contains the target graph.
    output_op: A `graphs.GraphsTuple` that contains the output graph.
    in_dict: The `dict` that contains testing results from the previous batch.
    Returns:
    in_dict: The updated `dict` that contains test results of the additional batch.
    """
    targets = target_gt.globals
    outputs = output_gt.globals
    labels, scores = [], []
    for t, o in zip(targets, outputs):
        in_dict[name+'_labels'].append(t.numpy().astype(np.float64).tolist())
        in_dict[name+'_scores'].append(tf.nn.softmax(o).numpy().astype(np.float64).tolist())
       
    return in_dict