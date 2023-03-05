import torch as th
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from tqdm import tqdm
import copy
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from matplotlib import colors

def get_npyimages(data_path, kappa, batch_size=100, shuffle_size_tr=1, limited_datasize=None, do_plot=False, make_Qk_image=True, n_class=6, extra_information=[]):
    if (limited_datasize > 0):
        datasize = limited_datasize
    else:
        with open(data_path.split('.npy')[0] + '.count') as f:
            datasize = (int(f.readline()))
    
    inputs_x = np.zeros((datasize, 2,75,75), dtype='float32')
    if make_Qk_image:
        inputs_x1 = np.zeros((datasize, 200, 75, 75), dtype='float32') #hQ_list
    else:
        inputs_x1 = np.zeros((datasize, 1, 75, 75))
    inputs_x2 = np.zeros((datasize,200), dtype='float32') #pTnorm_list #//set initially to be -1 so the user can check this easily

    inputs_y = np.zeros((datasize,n_class), dtype='float32')
    maxN_constituents = 0

    if (len(extra_information)>0):
        print ("There are ", len(extra_information)," extra parameter(s) are also read and return into the last return variable" )
        extra_inputs = np.zeros((datasize, len(extra_information)))


    imags = [np.zeros((75,75)), np.zeros((75,75))]
    fig1, ax1 = plt.subplots(3,3, figsize=(12,10))
    fig2, ax2 = plt.subplots(3,3, figsize=(12,10))
    figure_i = 0
    with open(data_path, 'rb') as npy_file:
        with tqdm(total=datasize, ncols=50) as pbar:
            for i in range(datasize):   
                # print (a+'\n------------------------------------')
                # #print (a)
                a = np.load(npy_file, allow_pickle=True)

                N_constituents = len(a)
                maxN_constituents = max(N_constituents, maxN_constituents)
                inputs_x[i][0] = np.array(a[1]) #hpT
                if (make_Qk_image):
                    hQ_list = np.array(a[2]) #hQ
                    pTnorm_list = a[3]
                    inputs_x[i][1] = np.sum(hQ_list * (pTnorm_list.reshape(len(hQ_list),1,1))**kappa, axis=0)
                else:
                    inputs_x[i][1] = np.array(a[2])

                inputs_y[i] = np.array(a[0]) #labels
                
                if len(extra_information) > 0:
                    for iii in range(len(extra_information)):
                        extra_inputs[i][iii] = a[iii+3]
                pbar.update(1)

                if do_plot == True:
                    if (inputs_y[i][0] != 1):#only record W+/W+
                        continue
                    imags[0] += copy.deepcopy(inputs_x[i][0].astype(float))
                    imags[1] += copy.deepcopy(inputs_x[i][1].astype(float))
                    if int(figure_i/3)>=3:
                        continue
                    axn1 = ax1[int(figure_i/3),int(figure_i)%3]
                    axn2 = ax2[int(figure_i/3),int(figure_i)%3]
                    imag1 = axn1.imshow(inputs_x[i][0], extent=[-np.pi, np.pi, -5, 5], norm=colors.SymLogNorm(linthresh=0.1, linscale=0.4,vmin=0), cmap='Blues')
                    imag2 = axn2.imshow(inputs_x[i][1]*len(hQ_list)**2, extent=[-np.pi, np.pi, -5, 5], interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
                    plt.colorbar(imag1, ax=axn1, shrink=0.9)
                    plt.colorbar(imag2, ax=axn2, shrink=0.9)
                    axn1.set_xlabel(r'$\phi^\prime$', fontsize=14)
                    axn1.set_ylabel(r'$\eta$', fontsize=14)
                    axn1.set_aspect(1./axn1.get_data_ratio(), adjustable='box')
                    axn2.set_xlabel(r'$\phi^\prime$', fontsize=14)
                    axn2.set_ylabel(r'$\eta$', fontsize=14)
                    axn2.set_aspect(1./axn2.get_data_ratio(), adjustable='box')
                    figure_i += 1

    if do_plot == True:
        plt.tight_layout()
        fig2.savefig('figures/Qk_jet-events.png', dpi=300)
        fig1.savefig('figures/pT_jet-events.png', dpi=300)
        plt.close()
        fig, ax = plt.subplots(1,1, figsize=(12,10))
        imag = ax.imshow(imags[1], extent=[-np.pi, np.pi, -5, 5], interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
        plt.colorbar(imag, ax=ax, shrink=0.9)
        ax.set_xlabel(r'$\phi^\prime$', fontsize=14)
        ax.set_ylabel(r'$\eta$', fontsize=14)
        ax.set_aspect(1./axn2.get_data_ratio(), adjustable='box')
        fig.savefig('figures/Qk_jet-average.png', dpi=300)
        plt.close()

    #print (maxN_constituents)
    #print (inputs_x.dtype, inputs_y.dtype)
    #print (inputs_x.shape, inputs_y.shape)
    N_labels = len(inputs_y[0])
    inputs_x = th.from_numpy(inputs_x)
    inputs_x1 = th.from_numpy(inputs_x1)
    inputs_x2 = th.from_numpy(inputs_x2)
    inputs_y = th.from_numpy(inputs_y)
    dataset = TensorDataset(inputs_x, inputs_x1, inputs_x2, inputs_y) # create your datset
    del inputs_x, inputs_x1, inputs_x2, inputs_y
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_size_tr, num_workers=0) # create your dataloader
    print ("The number of batches in data:", len(data_loader))
    del dataset
    if len(extra_information)>0:
        return (data_loader, N_labels, datasize, extra_inputs)
    else:
        return (data_loader, N_labels, datasize)
    


def get_npydataset(data_path, batch_size=100, shuffle_size_tr=1, limited_datasize=None, n_class=6):
    if (limited_datasize >0):
        datasize = limited_datasize
    else:
        with open(data_path.split('.npy')[0] + '.count') as f:
            datasize = (int(f.readline()))
    
    inputs_x = np.zeros((datasize, 2,75,75), dtype='float32')
    inputs_x1 = np.zeros((datasize, 200, 75, 75), dtype='float32') #hQ_list
    inputs_x2 = np.zeros((datasize,200), dtype='float32') #pTnorm_list

    inputs_y = np.zeros((datasize,n_class), dtype='float32')
    maxN_constituents = 0
    with open(data_path, 'rb') as npy_file:
        with tqdm(total=datasize, ncols=50) as pbar:
            for i in range(datasize):   
                # print (a+'\n------------------------------------')
                # #print (a)
                a = np.load(npy_file, allow_pickle=True)
                pTnorm_list = a[3]

                N_constituents = len(a)
                maxN_constituents = max(N_constituents, maxN_constituents)
                inputs_x[i][0] = np.array(a[1]) #hpT
                hQ_list = np.array(a[2]) #hQ
                inputs_x1[i] = np.concatenate((hQ_list.astype('float32'), np.zeros(((np.array(inputs_x1[i].shape) - np.array(hQ_list.shape))[0], 75, 75))), axis=0) #Q_list
                inputs_x2[i] = np.concatenate((pTnorm_list.astype('float32'), np.zeros(((np.array(inputs_x2[i].shape) - np.array(pTnorm_list.shape))[0]))), axis=0) #pTnorm_list

                inputs_y[i] = np.array(a[0]) #labels

                pbar.update(1)
    del a

    #print (maxN_constituents)
    #print (inputs_x.dtype, inputs_y.dtype)
    #print (inputs_x.shape, inputs_y.shape)
    N_labels = len(inputs_y[0])
    if (N_labels !=  n_class):
        AssertionError("Error: The class set in the data is not the same as the one set in the script.")
    inputs_x = th.from_numpy(inputs_x)
    inputs_x1 = th.from_numpy(inputs_x1)
    inputs_x2 = th.from_numpy(inputs_x2)
    inputs_y = th.from_numpy(inputs_y)
    dataset = TensorDataset(inputs_x, inputs_x1, inputs_x2, inputs_y) # create your datset
    del inputs_x, inputs_x1, inputs_x2, inputs_y
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_size_tr) # create your dataloader
    print ("The number of batches in data:", len(data_loader))
    del dataset

    return (data_loader, N_labels, datasize)