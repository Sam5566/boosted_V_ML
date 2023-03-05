from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import ijson
from tfr_utils import *
from tqdm import tqdm
from contextlib import ExitStack
import itertools
import os

__all__ = ['create_TFRecord', 'create_npy']

def list_feature(labels):
	""" Create feature for list of features"""
	return _list_float_feature(labels)

def image_feature(image):
    """ Create feature for images """
    return _list_of_lists_float_feature(image)

def get_sequence_example_object(data_element_dict):
    """ Creates a SequenceExample object from a dictionary for a single data element 
    data_element_dict is a dictionary for each element in .json file created by the fastjet code. 
    """
    # Context contains all scalar and list features
    context = tf.train.Features(
            feature=
            {
                'labels'  : list_feature(data_element_dict['labels']),
            }
    )
    
    # Feature_lists contains all lists of lists
    feature_lists = tf.train.FeatureLists(
            feature_list=
            {
                'pT'   : image_feature(data_element_dict['pT']),
                'Qk'   : image_feature(data_element_dict['Qk']),
            }
    )
                
    sequence_example = tf.train.SequenceExample(context       = context,
                                                feature_lists = feature_lists)
    
    return sequence_example

def determine_entry_type(label_list, N_class):
    onehot_dict = {}
    for iii in range(N_class):
        onehot_list = [0]*N_class
        onehot_list[iii] += 1
        #print (type(str(label_list[iii])))
        onehot_dict[str(label_list[iii])] = np.array(onehot_list)
        # permutation of labeling
        tmp = str(label_list[iii])
        #print (tmp)
        #print (tmp.split('/')[1]+'/'+tmp.split('/')[0])
        try:
            onehot_dict[tmp.split('/')[1]+'/'+tmp.split('/')[0]] = np.array(onehot_list)
        except:
            print (tmp+' has no permutation.')
    return onehot_dict

def determine_entry2(onehot_dict, entry, idx):
    try:
        return onehot_dict[str(entry[0])]
    except:
        tmp = str(entry[0])
        print ("not string entry, entry[0] is", entry[0], "at", idx)

def determine_entry(entry, idx): #// no use anymore

    if (entry[0]=='Z/Z'):
        #print ("Z")
        entry[0]=[0,0,1,0,0,0]
        #N_Z += 1
    elif (entry[0]=='W+/W+'):
        #print ("W+")
        entry[0]=[1,0,0,0,0,0]
        #N_Wp += 1
    elif (entry[0]=='W-/W-'):
        #print ("W-")
        entry[0]=[0,1,0,0,0,0]
        #N_Wm += 1
    elif (entry[0]=='W+/W-') or (entry[0]=='W-/W+'):
        #print ("W-")
        entry[0]=[0,0,0,1,0,0]
        #N_Wm += 1
    elif (entry[0]=='W+/Z') or (entry[0]=='Z/W+'):
        #print ("W-")
        entry[0]=[0,0,0,0,1,0]
        #N_Wm += 1
    elif (entry[0]=='Z/W-') or (entry[0]=='W-/Z'):
        #print ("W-")
        entry[0]=[0,0,0,0,0,1]
        #N_Wm += 1
    elif (entry[0]==[0,1,0,0,0,0]) or (entry[0]==[1,0,0,0,0,0]) or (entry[0]==[0,0,1,0,0,0]):
        pass
    else:
        print ("not string entry, entry[0] is", entry[0], "at", idx)
        raise AssertionError
    return entry[0]


def create_TFRecord(npy_files):
    datasizes = []

    NameOfDirectory = npy_files[0].split('/')[0] + '/' + '_and_'.join([string.split('/')[-1].split('.npy')[0] for string in npy_files])
    if type(npy_files)==str:
        with open(npy_files.split('.npy')[0] + '.count') as f:
            datasizes.append(int(f.readline()))
        npy_files = [npy_files]
    elif type(npy_files)==list:
        for fname in npy_files:
            print (fname.split('.npy')[0] + '.count')
            with open(fname.split('.npy')[0] + '.count') as f:
                datasizes.append(int(f.readline()))
        label_list = np.array([np.load(npy_file, allow_pickle=True) for npy_file in npy_files])[:,0]
    
    print ("datasizes in the npy", datasizes)
    #datasizes = [235000, 250000, 220000]
    #datasizes = [130000]*6
    #datasizes = [250000, 250000, 250000]
    #datasizes = [min(datasizes[0],250000), min(datasizes[1],250000), min(datasizes[2],250000)]
    for iii in range(len(datasizes)):
        datasizes[iii] = min(datasizes[iii], 10000)
    print ("redefine datasizes to", datasizes)
    
    datasize = sum(datasizes)
    trainsize = int(datasize*0.8)
    validsize = int(trainsize*0.2)
    testsize  = int(datasize-trainsize)
    trainsize = int(trainsize-validsize)
    
    idlist = np.array(list(itertools.chain.from_iterable([[idx]*datasizes[idx] for idx in range(len(datasizes))])), dtype=np.int64)
    np.random.shuffle(idlist)

    print ("Training, validation, and testing set are saved in "+NameOfDirectory)
    if not os.path.isdir(NameOfDirectory):
        os.system('mkdir '+NameOfDirectory)
    else:
        print ('directory already there.')
        #os.system('trash '+NameOfDirectory+'/*')
        os.system('ls '+NameOfDirectory)

    N_class = len(datasizes)
    print (str(label_list))
    onehot_dict = determine_entry_type(label_list, N_class)
    print ("The corresponding one-hot encoding for the sample", onehot_dict)

    with tqdm(total=datasize) as pbar:
        with ExitStack() as stack:
            npy_readers = [stack.enter_context(open(npy_file, 'rb')) for npy_file in npy_files]
            
            tr_list = idlist[:trainsize]
            vl_list = idlist[trainsize:trainsize+validsize]
            te_list = idlist[trainsize+validsize:]

            N_data = np.zeros((N_class), dtype='int') #N_Wp, N_Wm, N_Z
            with tf.io.TFRecordWriter(NameOfDirectory +'/train.tfrecord') as tfwriter:
                with open(NameOfDirectory +'/train.npy', 'wb') as imagewriter:
                    for idx in tr_list:
                        entry = np.load(npy_readers[idx], allow_pickle=True)
                        #entry[0] = determine_entry(entry, idx)
                        entry[0] = determine_entry2(onehot_dict, entry, idx)
                        #entry[0] = np.array(entry[0])
                        dict_obj = {'labels': entry[0], 'pT': entry[1], 'Qk': entry[2]}
                        sequence_example = get_sequence_example_object(dict_obj)
                        tfwriter.write(sequence_example.SerializeToString())
                        
                        np.save(imagewriter, entry)
                        N_data += entry[0]
                        #pbar.set_description(f'Train (Number of data: {np.sum(N_data):d}) W+W+:[{N_data[0]:d}] | W-W-:[{N_data[1]:d}] | ZZ:[{N_data[2]:d}] | W+W-:[{N_data[3]:d}] | W+Z:[{N_data[4]:d}] | W-Z:[{N_data[5]:d}] |')
                        pbar.set_description(f'Train (Number of data: {np.sum(N_data):d})')
                        pbar.update(1)
            print ("\n---------------\n")
            N_data = np.zeros((N_class), dtype='int') #N_Wp, N_Wm, N_Z
            with tf.io.TFRecordWriter(NameOfDirectory +'/valid.tfrecord') as tfwriter:
                with open(NameOfDirectory +'/valid.npy', 'wb') as imagewriter:
                    for idx in vl_list:
                        entry = np.load(npy_readers[idx], allow_pickle=True)
                        #entry[0] = determine_entry(entry, idx)
                        entry[0] = determine_entry2(onehot_dict, entry, idx)
                        #entry[0] = np.array(entry[0])
                        dict_obj = {'labels': entry[0], 'pT': entry[1], 'Qk': entry[2]}
                        sequence_example = get_sequence_example_object(dict_obj)
                        tfwriter.write(sequence_example.SerializeToString())
                        
                        np.save(imagewriter, entry)
                        N_data += entry[0]
                        #pbar.set_description(f'Valid (Number of data: {np.sum(N_data):d}) W+W+:[{N_data[0]:d}] | W-W-:[{N_data[1]:d}] | ZZ:[{N_data[2]:d}] | W+W-:[{N_data[3]:d}] | W+Z:[{N_data[4]:d}] | W-Z:[{N_data[5]:d}] |')
                        pbar.set_description(f'Valid (Number of data: {np.sum(N_data):d})')
                        pbar.update(1)
            print ("\n---------------\n")
            N_data = np.zeros((N_class), dtype='int') #N_Wp, N_Wm, N_Z
            with tf.io.TFRecordWriter(NameOfDirectory +'/test.tfrecord') as tfwriter:
                with open(NameOfDirectory +'/test.npy', 'wb') as imagewriter:
                    for idx in te_list:
                        entry = np.load(npy_readers[idx], allow_pickle=True)
                        
                        #entry[0] = determine_entry(entry, idx)
                        entry[0] = determine_entry2(onehot_dict, entry, idx)
                        #entry[0] = np.array(entry[0])
                        dict_obj = {'labels': entry[0], 'pT': entry[1], 'Qk': entry[2]}
                        sequence_example = get_sequence_example_object(dict_obj)
                        tfwriter.write(sequence_example.SerializeToString())
                        
                        np.save(imagewriter, entry)
                        N_data += entry[0]
                        #pbar.set_description(f'Test (Number of data: {np.sum(N_data):d}) W+W+:[{N_data[0]:d}] | W-W-:[{N_data[1]:d}] | ZZ:[{N_data[2]:d}] | W+W-:[{N_data[3]:d}] | W+Z:[{N_data[4]:d}] | W-Z:[{N_data[5]:d}] |')
                        pbar.set_description(f'Test (Number of data: {np.sum(N_data):d})')
                        pbar.update(1)
 
    with open(NameOfDirectory +'/train.count', 'w+') as f:
        f.write('{0:d}\n'.format(trainsize))
    with open(NameOfDirectory +'/valid.count', 'w+') as f:
        f.write('{0:d}\n'.format(validsize))
    with open(NameOfDirectory +'/test.count', 'w+') as f:
        f.write('{0:d}\n'.format(testsize))
        
    return idlist



def create_npy(npy_files, set_idlist=None):
    datasizes = []

    NameOfDirectory = npy_files[0].split('/')[0] + '/' + '_and_'.join([string.split('/')[-1].split('.npy')[0] for string in npy_files])
    if type(npy_files)==str:
        with open(npy_files.split('.npy')[0] + '.count') as f:
            datasizes.append(int(f.readline()))
        npy_files = [npy_files]
    elif type(npy_files)==list:
        for fname in npy_files:
            print (fname.split('.npy')[0] + '.count')
            with open(fname.split('.npy')[0] + '.count') as f:
                datasizes.append(int(f.readline()))
        dataset = np.array([np.load(npy_file, allow_pickle=True) for npy_file in npy_files])
    
    print ("datasizes in the npy", datasizes)
    #datasizes = [235000, 250000, 220000]
    #datasizes = [130000]*6
    #datasizes = [250000, 250000, 250000]
    for iii in range(len(datasizes)):
        datasizes[iii] = min(datasizes[iii], 25000)
    print ("redefine datasizes to", datasizes)
    
    datasize = sum(datasizes)
    trainsize = int(datasize*0.8)
    validsize = int(trainsize*0.2)
    testsize  = int(datasize-trainsize)
    trainsize = int(trainsize-validsize)

    
    if (set_idlist is None):
        idlist = np.array(list(itertools.chain.from_iterable([[idx]*datasizes[idx] for idx in range(len(datasizes))])), dtype=np.int64)
        np.random.shuffle(idlist)
    else:
        idlist = set_idlist

    print ("Training, validation, and testing set are saved in "+NameOfDirectory)
    if not os.path.isdir(NameOfDirectory):
        os.system('mkdir '+NameOfDirectory)
    else:
        print ('directory already there.')
        #os.system('trash '+NameOfDirectory+'/*')
        os.system('ls '+NameOfDirectory)

    with tqdm(total=datasize) as pbar:
        with ExitStack() as stack:
            npy_readers = [stack.enter_context(open(npy_file.split('.npy')[0]+'2.npy', 'rb')) for npy_file in npy_files]
            
            tr_list = idlist[:trainsize]
            vl_list = idlist[trainsize:trainsize+validsize]
            te_list = idlist[trainsize+validsize:]

            N_Z, N_Wp, N_Wm = 0, 0, 0
            #npy_sample = []
            with open(NameOfDirectory +'/train2.npy', 'wb') as imagewriter:
                for idx in tr_list:
                    entry = np.load(npy_readers[idx], allow_pickle=True)
                    #npy_sample.append(entry)
                    np.save(imagewriter, entry)
                    pbar.update(1)
            #npy_sample = np.array(npy_sample)
            #np.save(NameOfDirectory +'/train2.npy', npy_sample)
            #print (N_Z, N_Wp, N_Wm)

            N_Z, N_Wp, N_Wm = 0, 0, 0
            #npy_sample = []
            with open(NameOfDirectory +'/valid2.npy', 'wb') as imagewriter:
                for idx in vl_list:
                    entry = np.load(npy_readers[idx], allow_pickle=True)
                    #npy_sample.append(entry)
                    np.save(imagewriter, entry)
                    pbar.update(1)
            #npy_sample = np.array(npy_sample)
            #np.save(NameOfDirectory +'/valid2.npy', npy_sample)
            #print (N_Z, N_Wp, N_Wm)
            
            N_Z, N_Wp, N_Wm = 0, 0, 0
            #npy_sample = []
            with open(NameOfDirectory +'/test2.npy', 'wb') as imagewriter:
                for idx in te_list:
                    entry = np.load(npy_readers[idx], allow_pickle=True)
                    #npy_sample.append(entry)
                    np.save(imagewriter, entry)
                    pbar.update(1)
            #npy_sample = np.array(npy_sample)
            #np.save(NameOfDirectory +'/test2.npy', npy_sample)
            #print (N_Z, N_Wp, N_Wm)
 
    with open(NameOfDirectory +'/train2.count', 'w+') as f:
        f.write('{0:d}\n'.format(trainsize))
    with open(NameOfDirectory +'/valid2.count', 'w+') as f:
        f.write('{0:d}\n'.format(validsize))
    with open(NameOfDirectory +'/test2.count', 'w+') as f:
        f.write('{0:d}\n'.format(testsize))
