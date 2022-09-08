import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from tqdm import tqdm
from contextlib import ExitStack
import itertools

dataname = 'sample/samples_kappa0.3/samples_kappa0.3'
data_count = int(np.loadtxt(dataname+'.count'))
print (data_count)

i=0
labels=[]
pT = []
Qk = []
figure_data=False
print ('Start collecting data...')
if (figure_data):
    with tqdm(data_count) as pbar:
        with open(dataname+'.npy', 'rb') as npy_file:
            for i in range(data_count):
                aaa = (np.load(npy_file, allow_pickle=True))
                labels.append(aaa[0])
                pT.append(aaa[1])
                Qk.append(aaa[2])
                pbar.update(1)

    print ("sample shape:",np.shape(pT))
    sample1 = np.array(pT)
    sample2 = np.array(Qk)
    labels  = np.array(labels)
    print (labels)
    labels_num = np.where(labels=='W+', 1, labels)
    labels_num = np.where(labels_num=='W-', 2, labels_num)
    labels_num = np.where(labels=='Z', 0, labels_num)
    labels_num = labels_num.astype(int)
else:
    df = pd.read_csv(dataname+'_properties.txt', index_col=0).replace('W+',1).replace('W-',2).replace('Z',0)
    print (df)

df = df.sample(frac=1)
#X, testX, Y, testY = train_test_split(sample1.reshape(sample1.shape[0], sample1.shape[1]*sample1.shape[2]), labels_num, test_size = 0.3, random_state=1)
trainX, testX, trainY, testY = train_test_split(df.drop(['particle_type'], axis=1), df['particle_type'], test_size = 0.15, random_state=1)
#print (X.shape, Y.shape)
#trainX, valX, trainY, valY = train_test_split(X, Y, test_size = 0.15, random_state=1)
print (trainX.shape, trainY.shape)

clf = GradientBoostingClassifier().fit(trainX, trainY)

print (clf.score(testX, testY))

predY = clf.predict(testX)
confusion = confusion_matrix(testY, predY)
print('Confusion Matrix:')
print(confusion)

testX['prediction'] = predY

plt.scatter(testX[testX['prediction']==0]['jet charge'], testX[testX['prediction']==0]['jet mass'], c='r', label=r'$Z$')
plt.scatter(testX[testX['prediction']==1]['jet charge'], testX[testX['prediction']==1]['jet mass'], c='b', label=r'$W^+$')
plt.scatter(testX[testX['prediction']==2]['jet charge'], testX[testX['prediction']==2]['jet mass'], c='g', label=r'$W^-$')
plt.legend(loc='upper left')
plt.xlabel(r'$Q_k$', fontsize=12)
plt.ylabel(r'$M$ [GeV]', fontsize=12)
plt.title(r'SKLEARN ($\kappa=0.3$)', fontsize=15)
plt.legend(loc='upper left')
plt.savefig('figures/BDT-sklearn_prediction_kappa0.3.png', dpi=300)
