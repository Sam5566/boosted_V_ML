import numpy as np
#import pandas as pd
import sys
import os

def shuffle_along_axis(a, axis):
    #idx = np.random.rand(*a.shape).argsort(axis=axis)
    #print (idx)
    idx_labels = (np.arange(len(a)))
    #print (idx_labels)
    np.random.shuffle(idx_labels)
    #print (idx_labels)
    idx = np.array([])
    idx = np.append(idx,idx_labels)
    idx = np.append(idx,idx_labels)
    idx = np.append(idx,idx_labels)
    idx = (idx.reshape(len(a[0]),len(a))).astype(int).T
    print (idx)
    return np.take_along_axis(a,idx,axis=axis)

## Data reading
def npy_shuffle(npy_files):
    print ('Shuffling data ...')
    datasizes = []
    df = []
    if type(npy_files)==str:
        with open(npy_files.split('.npy')[0] + '.count') as f:
            datasizes.append(int(f.readline()))
        npy_files = [npy_files]
        dataset = np.load(npy_files, allow_pickle=True)
    elif type(npy_files)==list:
        for fname in npy_files:
            with open(fname.split('.npy')[0] + '.count') as f:
                datasizes.append(int(f.readline()))
        dataset = np.array([np.load(npy_file, allow_pickle=True) for npy_file in npy_files])
    if len(datasizes)>1:
        raise AssertionError("We do not input multiple datasets")
    dataset = []
    Nmax = 10000000000
    i_1 = 0
    i_2 = 0
    i_3 = 0
    print (datasizes[0])
    data_type = 'ternary'
    if (i_1==Nmax):
        data_type = 'binary-Wpm'
    elif (i_2==Nmax):
        data_type = 'binary-ZWm'
    elif (i_3==Nmax):
        data_type = 'binary-ZWp'
    with open(npy_files[0], 'rb') as datas_file:
        print (datas_file)
        for i in range(datasizes[0]):
            print (i)
            try:
                aaaaaa = (np.load(datas_file, allow_pickle=True))
            except:
                print ("The data is not pickle or the data length is not as same as the one in count file.")
            if (i_1<Nmax) and ((aaaaaa[0]=='Z') or (aaaaaa[0]==[0,0,1])):
                i_1+=1
            elif (i_2<Nmax) and ((aaaaaa[0]=='W+') or (aaaaaa[0]==[1,0,0])):
                i_2+=1
            elif (i_3<Nmax) and ((aaaaaa[0]=='W-') or (aaaaaa[0]==[0,1,0])):
                i_3+=1
            elif (i_3==Nmax and i_2==Nmax and i_1==Nmax):
                break
            else:
                continue
            dataset.append(aaaaaa)
    # with open(npy_files[0], 'rb') as datas:
    #    for i in range(datasizes[0]):
    #        aaaaaa = (np.load(datas, allow_pickle=True))
    #        dataset.append(aaaaaa)
        dataset = np.array(dataset)

    print ((dataset.shape))
    print (len(dataset))
    #print (dataset[-10:-1,0])
    
    print ("################")
    """
    print (dataset[:,0])
    abc = (dataset[:,1][0])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][1])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][2])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][3])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][4])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][5])
    print (abc[abc!=0][:2])
    """
    dataset = shuffle_along_axis(dataset, axis=0)
    """
    #print (dataset[-10:-1,0])
    print (dataset[:,0])
    abc = (dataset[:,1][0])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][1])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][2])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][3])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][4])
    print (abc[abc!=0][:2])
    abc = (dataset[:,1][5])
    print (abc[abc!=0][:2])
    exit()
    """

    NameOfDirectory = '_and_'.join([string.split('/')[0] for string in npy_files])+data_type
    print (NameOfDirectory)
    if not os.path.isdir(NameOfDirectory):
        os.system('mkdir '+NameOfDirectory)
    else:
        print ('directory already there.')
        os.system('ls '+NameOfDirectory)
    
    with open(NameOfDirectory+'/'+npy_files[0].split('/')[0]+'_shuffled.npy', 'wb') as imagewriter:
        for data in dataset:
            np.save(imagewriter, data)
    return i_1+i_2+i_3, NameOfDirectory+'/'+npy_files[0].split('/')[0]+'_shuffled.npy'
    
