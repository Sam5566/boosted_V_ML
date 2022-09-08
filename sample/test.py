import numpy as np
with open('samples_kappa0.15.npy', 'rb') as datas:
    for i in range(636414):
        a = np.load(datas, allow_pickle=True)
        if a[0]!='Z':
            print (a[0])

