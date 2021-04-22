import pandas as pd
from tqdm import tqdm_notebook

import numpy as np
from sklearn.model_selection import KFold

prefix = 'data/tr/'

train_df = pd.read_csv(prefix + '10-fold.csv', header=None)


X = np.array(train_df[1])
y = np.array(train_df[0])

kfold = KFold(10, False) # no shuffling

fold = 0
for train_idx, test_idx in kfold.split(X):
    #print(train_idx, test_idx)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]    
    
    train_df = pd.DataFrame({
        'id':range(len(X_train)), #range(len(test_df)),
        'label':y_train, #train_df[0],
        'alpha':['a']*X_train.shape[0], #test_df.shape[0],
        #'text':np_f.replace(X_train, '\n', ' ')
        'text': X_train#.replace(r'\n', ' ', regex=True) #test_df[1].replace(r'\n', ' ', regex=True)
    })


    dev_df = pd.DataFrame({
        'id':range(len(X_test)),
        'label':y_test,
        'alpha':['a']*X_test.shape[0],
        #'text':np_f.replace(X_test, '\n', ' ')
        'text': X_test#.replace(r'\n', ' ', regex=True)
    })
    

    train_df.to_csv(prefix + '10-fold/' + str(fold) + '_train.tsv', sep='\t', index=False, header=False, columns=train_df.columns)
    dev_df.to_csv(prefix + '10-fold/' + str(fold) + '_test.tsv', sep='\t', index=False, header=False, columns=dev_df.columns)
    
    fold += 1