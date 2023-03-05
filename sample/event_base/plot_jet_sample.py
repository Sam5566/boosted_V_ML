import sys
import os
#os.chdir('/home/samhuang/ML/sample')
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

histranges = [-np.pi, np.pi, -5, 5]

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
    outputfiledir = sys.argv[i].split('/')[0]+'/'+ sys.argv[i].split('/')[1]+'/'+ sys.argv[i].split('/')[2]+'/'+ sys.argv[i].split('/')[3]+'/' + sys.argv[i].split('/')[4]+'/event_base/' + "samples_kappa"+str(kappa)+'/'
    imagename = outputfiledir + inname + '.npy'
    countname = outputfiledir + inname + '.count'
    print (imagename)

    dfs.append(pd.read_csv(outputfiledir + inname + '_properties.txt'))
    #print (dfs[i-2])
    fig1, ax1 = plt.subplots(3,3, figsize=(12,10))
    fig2, ax2 = plt.subplots(3,3, figsize=(12,10))
    with open(imagename, 'rb') as datafile:
        for j in range(len(dfs[i-2])):
        #for j in range(100):
            entry = np.load(datafile, allow_pickle=True)
            #print (type(entry[1]))
            #print (np.max(entry[1]))
            imags[i-2][0] += copy.deepcopy(entry[1].astype(float))
            imags[i-2][1] += copy.deepcopy(entry[2].astype(float))  
            #print (np.max(imags[i-2][0]))
            
            if int(j/3)>=3:
                continue
            axn1 = ax1[int(j/3),int(j)%3]
            axn2 = ax2[int(j/3),int(j)%3]
            imag1 = axn1.imshow(entry[1].astype(float), extent=histranges, norm=colors.SymLogNorm(linthresh=0.1, linscale=0.4,vmin=0), cmap='Blues')
            imag2 = axn2.imshow(entry[2].astype(float), extent=histranges, interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
            plt.colorbar(imag1, ax=axn1, shrink=0.9)
            plt.colorbar(imag2, ax=axn2, shrink=0.9)
            #axn.set_title(r'$\langle W^+W^+\rangle: p_T$ channel', fontsize=fontsize)
            axn1.set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
            axn1.set_ylabel(r'$\eta$', fontsize=fontsize)
            axn1.set_aspect(1./axn1.get_data_ratio(), adjustable='box')
            axn2.set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
            axn2.set_ylabel(r'$\eta$', fontsize=fontsize)
            axn2.set_aspect(1./axn2.get_data_ratio(), adjustable='box')

    plt.tight_layout()
    fig2.savefig('figures/'+ figure_folder + 'Qk_jet-'+str(i)+'.png', dpi=300)
    fig1.savefig('figures/'+ figure_folder + 'pT_jet-'+str(i)+'.png', dpi=300)
    plt.close()

#print (imags[0][0])
#print (type(imags[0][0]))

print ("Start plotting...")
## P_T image distribution
fig, ax = plt.subplots(2,3, figsize=(15,10))
imag = ax[0,0].imshow(imags[0][0]/len(dfs[0]), extent=histranges, norm=colors.SymLogNorm(linthresh=0.1, linscale=0.4,vmin=0), cmap='Blues')
plt.colorbar(imag, ax=ax[0,0], shrink=0.9)
ax[0,0].set_title(r'$\langle W^+W^+\rangle: p_T$ channel', fontsize=fontsize)
ax[0,0].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[0,0].set_ylabel(r'$\eta$', fontsize=fontsize)
ax[0,0].set_aspect(1./ax[0,0].get_data_ratio(), adjustable='box')

imag = ax[0,1].imshow(imags[2][0]/len(dfs[2]) - imags[0][0]/len(dfs[0]), extent=histranges, norm=colors.SymLogNorm(linthresh=0.1, linscale=0.6), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[0,1], shrink=0.9)
ax[0,1].set_title(r'$\langle Z Z\rangle-\langle W^+W^+\rangle: p_T$ channel', fontsize=fontsize)
ax[0,1].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[0,1].set_ylabel(r'$\eta$', fontsize=fontsize)
ax[0,1].set_aspect(1./ax[0,1].get_data_ratio(), adjustable='box')

imag = ax[0,2].imshow(imags[2][0]/len(dfs[2]), extent=histranges, norm=colors.SymLogNorm(linthresh=0.1, linscale=0.4,vmin=0), cmap='Blues')
plt.colorbar(imag, ax=ax[0,2], shrink=0.9)
ax[0,2].set_title(r'$\langle Z Z\rangle: p_T$ channel', fontsize=fontsize)
ax[0,2].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[0,2].set_ylabel(r'$\eta$', fontsize=fontsize)
ax[0,2].set_aspect(1./ax[0,2].get_data_ratio(), adjustable='box')

imag = ax[1,0].imshow(imags[3][0]/len(dfs[3]), extent=histranges, norm=colors.SymLogNorm(linthresh=0.1, linscale=0.4,vmin=0), cmap='Blues')
plt.colorbar(imag, ax=ax[1,0], shrink=0.9)
ax[1,0].set_title(r'$\langle W^+W^-\rangle: p_T$ channel', fontsize=fontsize)
ax[1,0].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[1,0].set_ylabel(r'$\eta$', fontsize=fontsize)
ax[1,0].set_aspect(1./ax[1,0].get_data_ratio(), adjustable='box')

imag = ax[1,1].imshow(imags[4][0]/len(dfs[4]) - imags[3][0]/len(dfs[3]), extent=histranges, norm=colors.SymLogNorm(linthresh=0.1, linscale=0.6), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[1,1], shrink=0.9)
ax[1,1].set_title(r'$\langle W^+ Z\rangle-\langle W^+W^-\rangle: p_T$ channel', fontsize=fontsize)
ax[1,1].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[1,1].set_ylabel(r'$\eta$', fontsize=fontsize)
ax[1,1].set_aspect(1./ax[1,1].get_data_ratio(), adjustable='box')

imag = ax[1,2].imshow(imags[4][0]/len(dfs[4]), extent=histranges, norm=colors.SymLogNorm(linthresh=0.1, linscale=0.4,vmin=0), cmap='Blues')
plt.colorbar(imag, ax=ax[1,2], shrink=0.9)
ax[1,2].set_title(r'$\langle W^+ Z \rangle: p_T$ channel', fontsize=fontsize)
ax[1,2].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[1,2].set_ylabel(r'$\eta$', fontsize=fontsize)
ax[1,2].set_aspect(1./ax[1,2].get_data_ratio(), adjustable='box')
plt.tight_layout()
plt.savefig('figures/'+ figure_folder + 'pT_jet.png', dpi=300)
plt.close()

## Q_k image distribution
fig, ax = plt.subplots(2,3, figsize=(15,10))
imag = ax[0,0].imshow(imags[0][1]/len(dfs[0]), extent=histranges, interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[0,0], shrink=0.9)
ax[0,0].set_title(r'$\langle W^+W^+\rangle: Q_k$ channel', fontsize=fontsize)
ax[0,0].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[0,0].set_ylabel(r'$\eta$', fontsize=fontsize)

imag = ax[0,1].imshow(imags[1][1]/len(dfs[1]), extent=histranges, interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[0,1], shrink=0.9)
ax[0,1].set_title(r'$\langle W^-W^-\rangle: Q_k$ channel', fontsize=fontsize)
ax[0,1].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[0,1].set_ylabel(r'$\eta$', fontsize=fontsize)

imag = ax[0,2].imshow(imags[2][1]/len(dfs[2]), extent=histranges, interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[0,2], shrink=0.9)
ax[0,2].set_title(r'$\langle ZZ \rangle: Q_k$ channel', fontsize=fontsize)
ax[0,2].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[0,2].set_ylabel(r'$\eta$', fontsize=fontsize)

imag = ax[1,0].imshow(imags[3][1]/len(dfs[3]), extent=histranges, interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[1,0], shrink=0.9)
ax[1,0].set_title(r'$\langle W^+W^-\rangle: Q_k$ channel', fontsize=fontsize)
ax[1,0].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[1,0].set_ylabel(r'$\eta$', fontsize=fontsize)

imag = ax[1,1].imshow(imags[4][1]/len(dfs[4]), extent=histranges, interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[1,1], shrink=0.9)
ax[1,1].set_title(r'$\langle W^+Z\rangle: Q_k$ channel', fontsize=fontsize)
ax[1,1].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[1,1].set_ylabel(r'$\eta$', fontsize=fontsize)

imag = ax[1,2].imshow(imags[5][1]/len(dfs[5]), extent=histranges, interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
plt.colorbar(imag, ax=ax[1,2], shrink=0.9)
ax[1,2].set_title(r'$\langle W^-Z \rangle: Q_k$ channel', fontsize=fontsize)
ax[1,2].set_xlabel(r'$\phi^\prime$', fontsize=fontsize)
ax[1,2].set_ylabel(r'$\eta$', fontsize=fontsize)
plt.tight_layout()
plt.savefig('figures/'+ figure_folder + 'Qk_jet.png', dpi=300)
plt.close()

## Q_k distribution
fig, ax = plt.subplots(1,1, figsize=(15,5))
ax.hist(dfs[0]['jet charge'], color='b', histtype='step', density=True, bins=100, range=[-2,2], label=r'$W^+/W^+$')
ax.hist(dfs[1]['jet charge'], color='r', histtype='step', density=True, bins=100, range=[-2,2], label=r'$W^-/W^-$')
ax.hist(dfs[2]['jet charge'], color='brown', histtype='step', density=True, bins=100, range=[-2,2], label=r'$Z/Z$')
ax.legend()
ax.set_xlim([-2,2])
ax.set_xlabel(r'$Q_k$')
ax.set_aspect(4)
plt.savefig('figures/'+ figure_folder + 'Qk_distribution.png', dpi=300)
plt.close()

## P_T distribution
fig, ax = plt.subplots(1,1, figsize=(15,5))
ax.hist(dfs[0]['jet mass'], color='b', histtype='step', density=True, bins=100, range=[60,120], label=r'$W^+/W^+$')
ax.hist(dfs[1]['jet mass'], color='r', histtype='step', density=True, bins=100, range=[60,120], label=r'$W^-/W^-$')
ax.hist(dfs[2]['jet mass'], color='brown', histtype='step', density=True, bins=100, range=[60,120], label=r'$Z/Z$')
ax.legend()
ax.set_xlim([60,120])
ax.set_xlabel(r'$p_T$')
ax.set_aspect(800)
plt.savefig('figures/'+ figure_folder + 'pT_distribution.png', dpi=300)

print ("*Finish plotting*")
