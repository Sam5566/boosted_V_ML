import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from tqdm import tqdm
from contextlib import ExitStack
import itertools
kappa = "0.15"
datafolder = '../sample/event_base/samples_kappa'+kappa+'/'
#datafolder = '../sample/jet_base/samples_kappa'+kappa+'_2jet/'
#datafolder = '../sample/event_base/samples_kappa'+kappa+'/'
datanames = ['VBF_H5pp_ww_jjjj', 'VBF_H5mm_ww_jjjj', 'VBF_H5z_zz_jjjj']
datanames = ['VBF_H5pp_ww_jjjj', 'VBF_H5mm_ww_jjjj', 'VBF_H5z_zz_jjjj', 'VBF_H5z_ww_jjjj', 'VBF_H5p_wz_jjjj', 'VBF_H5m_wz_jjjj']
#datanames = ['VBF_H5pp_ww_jjjj', 'VBF_H5z_zz_jjjj']
#datanames = ['VBF_H5pp_ww_jjjj', 'VBF_H5mm_ww_jjjj']
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
        #df = pd.read_csv(datafolder+dataname+'_properties.txt', index_col=0).replace('W+',1).replace('W-',2).replace('Z',0)[0:250000]
        df = pd.read_csv(datafolder+dataname+'_properties.txt', index_col=0).replace('W+/W+',1).replace('W-/W-',2).replace('Z/Z',0).replace('W-/W+',3).replace('W+/W-',3).replace('W+/Z',4).replace('Z/W+',4).replace('W-/Z',5).replace('Z/W-',5)#[0:300000]
        print (df)
        dfs.append(df)

df = pd.concat(dfs)
df = df.sample(frac=1)
onehotencoder = OneHotEncoder()
#print (df['particle_type'])
#labelencoder = LabelEncoder()
#aaa = labelencoder.fit_transform(df['particle_type'])
#print (onehotencoder.fit_transform(aaa))
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
params = {'max_depth': 6, 'objective': 'multi:softmax', 'num_class': 3, 'verbosity': 1, 'num_class': 3, 'early_stopping_rounds': 10, 'tree_method':'gpu_hist'}

plst = list(params.items())
num_rounds = 5000 # 迭代次數
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

model = xgb.XGBClassifier(plst, verbosity=1)
model.fit(trainX, trainY)
#model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=10)
print ("best best_ntree_limit",model.best_ntree_limit)


predY = model.predict(testX)
pred_prob = model.predict_proba(testX)
print(predY.shape)
print ('predY[:2]', predY[:2])
print ('preds_prob[:2]', pred_prob[:2])
#predY=model.predict(xgb_test)
print (predY)
accuracy = accuracy_score(testY, predY)
print("Accuracy: %.4f%%" % (accuracy * 100.0))

confusion = confusion_matrix(testY, predY)
print('Confusion Matrix:')
print(confusion)


print (testX, predY)
testX['prediction'] = predY
#testX[['prediction_proba1','prediction_proba2','prediction_proba3']] = pred_prob
testX[['prediction_proba1','prediction_proba2','prediction_proba3', 'prediction_proba4','prediction_proba5','prediction_proba6']] = pred_prob
testX['particle_type'] = testY


onehotencoder = OneHotEncoder()
print (testY)
labels=onehotencoder.fit_transform(pd.DataFrame(testY)).toarray()
#labels=testY.toarray()
print (labels)
'''
plt.scatter(testX[testX['prediction']==0]['jet charge'], testX[testX['prediction']==0]['jet mass'], c='r', label=r'$Z$')
plt.scatter(testX[testX['prediction']==1]['jet charge'], testX[testX['prediction']==1]['jet mass'], c='b', label=r'$W^+$')
plt.scatter(testX[testX['prediction']==2]['jet charge'], testX[testX['prediction']==2]['jet mass'], c='g', label=r'$W^-$')
plt.legend(loc='upper left')
plt.xlabel(r'$Q_k$', fontsize=12)
plt.ylabel(r'$M$ [GeV]', fontsize=12)
plt.title(r'XGBoost ($\kappa='+kappa+'$)', fontsize=15)
plt.legend(loc='upper left')
plt.savefig('figures/BDT_prediction_kappa'+kappa+'.png', dpi=300)
'''
fpr, tpr, roc_auc = dict(), dict(), dict()
#signal=[r'$Z$',r'$W^+$',r'$W^-$']
signal=[r'$ZZ$',r'$W^+W^+$',r'$W^-W^-$', r'$W^+W^-$', r'$W^+Z$', r'$W^-Z$']
n_class = len(signal)

for i in range(n_class):
    aa = testX[testX['particle_type']==i]
    print(signal[i] + " Accuracy: %.4f%%" % (accuracy_score(aa['particle_type'], aa['prediction']) * 100.0))
    fpr[i], tpr[i], _ = roc_curve(labels[:,i], pred_prob[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig = plt.figure(figsize=(8,6))
for i in range(n_class):
    print ('{0} (auc = {1:.2f}%)'.format(signal[i], roc_auc[i]*100))
    plt.plot(fpr[i], tpr[i], label='{0} (auc = {1:.2f}%)'.format(signal[i], roc_auc[i]*100))
            
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right', fontsize=15)
plt.gcf().savefig('figures/roc_curve_kappa'+kappa+'.png')
