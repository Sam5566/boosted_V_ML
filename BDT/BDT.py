import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from tqdm import tqdm
from contextlib import ExitStack
import itertools
kappa = "0.15"
datafolder = 'sample/samples_kappa'+kappa+'/'
datanames = ['VBF_H5pp_ww_jjjj', 'VBF_H5mm_ww_jjjj', 'VBF_H5z_zz_jjjj']
data_counts = []
for dataname in datanames: 
    data_counts.append(int(np.loadtxt(datafolder+dataname+'.count')))
print (data_counts)

i=0
labels=[]
pT = []
Qk = []
figure_data=False
print ('Start collecting data...')
if (figure_data):
    for ii, dataname in enumerate(datanames):
        data_count = data_counts[ii]
        with tqdm(data_count) as pbar:
            with open(datafolder+dataname+'.npy', 'rb') as npy_file:
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
    labels_num = np.where(labels_num=='Z', 0, labels_num)
    labels_num = labels_num.astype(int)
else:
    dfs = []
    for dataname in datanames:
        df = pd.read_csv(datafolder+dataname+'_properties.txt', index_col=0).replace('W+',1).replace('W-',2).replace('Z',0)
        print (df)
        dfs.append(df)

df = pd.concat(dfs)
df = df.sample(frac=1)
#X, testX, Y, testY = train_test_split(sample1.reshape(sample1.shape[0], sample1.shape[1]*sample1.shape[2]), labels_num, test_size = 0.3, random_state=1)
X, testX, Y, testY = train_test_split(df.drop(['particle_type'], axis=1), df['particle_type'], test_size = 0.2, random_state=1)
print (X.shape, Y.shape)
trainX, valX, trainY, valY = train_test_split(X, Y, test_size = 0.2, random_state=1)
print (trainX.shape, trainY.shape)

xgb_val = xgb.DMatrix(valX,label=valY)
xgb_train = xgb.DMatrix(trainX, label=trainY)
xgb_test = xgb.DMatrix(testX, label=testY)

params = {
        'max_depth': 5,                 # the maximum depth of each tree
        'eta': 0.5,                     # the training step for each iteration
        'eta_decay': 0.9,
        'min_eta': 0.05,
        'silent': 0,                    # logging mode - quiet
        'objective': 'multi:softmax',   # multiclass classification using the softmax objective
        'num_class': 10,                 # the number of classes that exist in this datset
        'early_stopping_rounds': 10,
        'n_estimators': 1000
        }
params = {'max_depth': 6, 'objective': 'multi:softmax', 'num_class': 3}

plst = list(params.items())
num_rounds = 5000 # 迭代次數
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=10)
print ("best best_ntree_limit",model.best_ntree_limit)


preds = model.predict(xgb_test)
print(preds.shape)
print (preds[:2])
predY=model.predict(xgb_test)
print (predY)
accuracy = accuracy_score(testY, predY)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

confusion = confusion_matrix(testY, predY)
print('Confusion Matrix:')
print(confusion)


print (testX, predY)
testX['prediction'] = predY

plt.scatter(testX[testX['prediction']==0]['jet charge'], testX[testX['prediction']==0]['jet mass'], c='r', label=r'$Z$')
plt.scatter(testX[testX['prediction']==1]['jet charge'], testX[testX['prediction']==1]['jet mass'], c='b', label=r'$W^+$')
plt.scatter(testX[testX['prediction']==2]['jet charge'], testX[testX['prediction']==2]['jet mass'], c='g', label=r'$W^-$')
plt.legend(loc='upper left')
plt.xlabel(r'$Q_k$', fontsize=12)
plt.ylabel(r'$M$ [GeV]', fontsize=12)
plt.title(r'XGBoost ($\kappa='+kappa+'$)', fontsize=15)
plt.legend(loc='upper left')
plt.savefig('figures/BDT_prediction_kappa'+kappa+'.png', dpi=300)

fpr, tpr, roc_auc = dict(), dict(), dict()
signal=[r'$Z$',r'$W^+$',r'$W^-$']
n_class = len(signal)
for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(testY, predY, pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

fig = plt.figure(figsize=(8,6))
for i in range(n_class):
    print ('{0} (auc = {1:0.2f})'.format(signal[i], roc_auc[i]))
    plt.plot(fpr[i], tpr[i], label='{0} (auc = {1:0.2f})'.format(signal[i], roc_auc[i]))
            
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right', fontsize=15)
plt.gcf().savefig('roc_curve_kappa'+kappa+'.png')
