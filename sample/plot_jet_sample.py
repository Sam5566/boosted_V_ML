import sys
import os
os.chdir('/home/samhuang/ML/sample')
add_path = '/home/samhuang/ML/sample'
if add_path not in sys.path:
    sys.path.insert(0,add_path)
print(sys.path)
from tfr_utils import *
import ROOT as r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from matplotlib import colors
import tensorflow as tf
import copy

mpl.rcParams['figure.facecolor'] = 'white'
fontsize = 14
ticksize = 12

#print (sys.argv)
kappa = float(sys.argv[1])
figure_folder = sys.argv[1] + '_and_'.join((sys.argv[i+2]).split('/')[5] for i in range(len(sys.argv[2:]))) + '/'
print (figure_folder)
os.system('mkdir figures/'+figure_folder)

dfs = []
imags = [[np.zeros((75,75)), np.zeros((75,75))] for i in range(len(sys.argv)-2)]
for i in range(len(sys.argv)):
    if (i<=1):
        continue
    inname = sys.argv[i].split('/')[5] #// should be changed with different directory structure
    outputfiledir = sys.argv[i].split('/')[0]+'/'+ sys.argv[i].split('/')[1]+'/'+ sys.argv[i].split('/')[2]+'/'+ sys.argv[i].split('/')[3]+'/' + sys.argv[i].split('/')[4]+'/' + "samples_kappa"+str(kappa)+'/'
    imagename = outputfiledir + inname + '.npy'
    countname = outputfiledir + inname + '.count'

    dfs.append(pd.read_csv(outputfiledir + inname + '_properties.txt'))
    #print (dfs[i-2])
    with open(imagename, 'rb') as datafile:
        for j in range(len(dfs[i-2])):
        #for j in range(100):
            entry = np.load(datafile, allow_pickle=True)
            #print (type(entry[1]))
            #print (np.max(entry[1]))
            imags[i-2][0] += copy.deepcopy(entry[1].astype(float))
            imags[i-2][1] += copy.deepcopy(entry[2].astype(float))  
            #print (np.max(imags[i-2][0]))

    
#print (imags[0][0])
#print (type(imags[0][0]))

## P_T image distribution
fig, ax = plt.subplots(1,3, figsize=(15,5))
imag = ax[0].imshow(imags[0][0]/len(dfs[0]), extent=[-0.8, 0.8, -0.8, 0.8], norm=colors.SymLogNorm(linthresh=0.1, linscale=0.4,vmin=0), cmap='Blues')
plt.colorbar(imag, ax=ax[0], shrink=0.9)
ax[0].set_title(r'$\langle W^+\rangle: p_T$ channel', fontsize=fontsize)
ax[0].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[0].set_ylabel(r'$\eta^\prime$', fontsize=fontsize)

imag = ax[1].imshow(imags[2][0]/len(dfs[2]) - imags[0][0]/len(dfs[0]), extent=[-0.8, 0.8, -0.8, 0.8], norm=colors.SymLogNorm(linthresh=0.1, linscale=0.6), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[1], shrink=0.9)
ax[1].set_title(r'$\langle Z \rangle-\langle W^+\rangle: p_T$ channel', fontsize=fontsize)
ax[1].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[1].set_ylabel(r'$\eta^\prime$', fontsize=fontsize)

imag = ax[2].imshow(imags[2][0]/len(dfs[2]), extent=[-0.8, 0.8, -0.8, 0.8], norm=colors.SymLogNorm(linthresh=0.1, linscale=0.4,vmin=0), cmap='Blues')
plt.colorbar(imag, ax=ax[2], shrink=0.9)
ax[2].set_title(r'$\langle Z \rangle: p_T$ channel', fontsize=fontsize)
ax[2].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[2].set_ylabel(r'$\eta^\prime$', fontsize=fontsize)
plt.tight_layout()
plt.savefig('figures/'+ figure_folder + 'pT_jet.png', dpi=300)
plt.close()

## Q_k image distribution
fig, ax = plt.subplots(1,3, figsize=(15,5))
imag = ax[0].imshow(imags[0][1]/len(dfs[0]), extent=[-0.8, 0.8, -0.8, 0.8], interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[0], shrink=0.9)
ax[0].set_title(r'$\langle W^+\rangle: Q_k$ channel', fontsize=fontsize)
ax[0].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[0].set_ylabel(r'$\eta^\prime$', fontsize=fontsize)

imag = ax[1].imshow(imags[1][1]/len(dfs[1]), extent=[-0.8, 0.8, -0.8, 0.8], interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[1], shrink=0.9)
ax[1].set_title(r'$\langle W^-\rangle: Q_k$ channel', fontsize=fontsize)
ax[1].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[1].set_ylabel(r'$\eta^\prime$', fontsize=fontsize)

imag = ax[2].imshow(imags[2][1]/len(dfs[2]), extent=[-0.8, 0.8, -0.8, 0.8], interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[2], shrink=0.9)
ax[2].set_title(r'$\langle Z \rangle: Q_k$ channel', fontsize=fontsize)
ax[2].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[2].set_ylabel(r'$\eta^\prime$', fontsize=fontsize)
plt.tight_layout()
plt.savefig('figures/'+ figure_folder + 'Qk_jet.png', dpi=300)
plt.close()

## Q_k distribution
fig, ax = plt.subplots(1,1, figsize=(15,5))
ax.hist(dfs[0]['jet charge'], color='b', histtype='step', density=True, bins=100, range=[-2,2], label=r'$W^+$')
ax.hist(dfs[1]['jet charge'], color='r', histtype='step', density=True, bins=100, range=[-2,2], label=r'$W^-$')
ax.hist(dfs[2]['jet charge'], color='brown', histtype='step', density=True, bins=100, range=[-2,2], label=r'$Z$')
ax.legend()
ax.set_xlim([-2,2])
ax.set_xlabel(r'$Q_k$')
ax.set_aspect(4)
plt.savefig('figures/'+ figure_folder + 'Qk_distribution.png', dpi=300)
plt.close()

## P_T distribution
fig, ax = plt.subplots(1,1, figsize=(15,5))
ax.hist(dfs[0]['jet mass'], color='b', histtype='step', density=True, bins=100, range=[60,120], label=r'$W^+$')
ax.hist(dfs[1]['jet mass'], color='r', histtype='step', density=True, bins=100, range=[60,120], label=r'$W^-$')
ax.hist(dfs[2]['jet mass'], color='brown', histtype='step', density=True, bins=100, range=[60,120], label=r'$Z$')
ax.legend()
ax.set_xlim([60,120])
ax.set_xlabel(r'$p_T$')
ax.set_aspect(800)
plt.savefig('figures/'+ figure_folder + 'pT_distribution.png', dpi=300)

