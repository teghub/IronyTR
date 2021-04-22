#

# For citation, please check:
# https://github.com/teghub/IronyTR

# This code uses TPOT [https://epistasislab.github.io/tpot/] to optimize the parameters of the chosen method for your data set.
# You may need to install additional libraries for this code to work.
# You may need to create the missing files.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import SCORERS
from tpot import TPOTClassifier

# 0-1
cl = ['class']
# "bow"
bow_cols = [
'class' # This list should be the same size with the BoW vector, fill the columns accordingly.
]
# "basic"
bsc = ['word_token', 'pmcount_token', 'em_exists', 'em_pmcount', 'qm_exists', 'qm_pmcount', 'pem_exists', 'pem_pmcount', 'pqm_exists', 'pqm_pmcount', 'quote_exists', 'quote_pmcount', 'ellipsis_exists', 'ellipsis_pmcount', 'emo_exists', 'emo_token', 'interject_exists', 'repet_exists', 'boost_exists', 'caps_exists']
# "polarity"
plr = ['word_token', 'pmcount_token', 'em_exists', 'em_pmcount', 'qm_exists', 'qm_pmcount', 'pem_exists', 'pem_pmcount', 'pqm_exists', 'pqm_pmcount', 'quote_exists', 'quote_pmcount', 'ellipsis_exists', 'ellipsis_pmcount', 'emo_exists', 'emo_token', 'interject_exists', 'repet_exists', 'boost_exists', 'caps_exists', 'polsum_token', 'pos_sum_token', 'neg_sum_token', 'max_pol', 'min_pol', 'max_min_diff', 'possum_negsum_diff', 'clash']
# "graph"
grp = ['word_token', 'pmcount_token', 'em_exists', 'em_pmcount', 'qm_exists', 'qm_pmcount', 'pem_exists', 'pem_pmcount', 'pqm_exists', 'pqm_pmcount', 'quote_exists', 'quote_pmcount', 'ellipsis_exists', 'ellipsis_pmcount', 'emo_exists', 'emo_token', 'interject_exists', 'repet_exists', 'boost_exists', 'caps_exists', 'i_graph', 'ni_graph']
# "polarity-graph"
pg = ['word_token', 'pmcount_token', 'em_exists', 'em_pmcount', 'qm_exists', 'qm_pmcount', 'pem_exists', 'pem_pmcount', 'pqm_exists', 'pqm_pmcount', 'quote_exists', 'quote_pmcount', 'ellipsis_exists', 'ellipsis_pmcount', 'emo_exists', 'emo_token', 'interject_exists', 'repet_exists', 'boost_exists', 'caps_exists', 'polsum_token', 'pos_sum_token', 'neg_sum_token', 'max_pol', 'min_pol', 'max_min_diff', 'possum_negsum_diff', 'clash', 'i_graph', 'ni_graph']

print("Select the feature set (all features are combined with BoW) \n Type one of the following (bow/basic/polarity/graph/polarity-graph):")
file = input()
print("Select algorithm \n Type one of the following (knn/svm/nb/dt/rf):")
limit = input()
print("Shuffled?(y/n):")
shuf = input()
print("Optimize by? \n Type one of the following (f1/accuracy):")
opt = input()

data = pd.read_csv('../data/features/' + file + '-features-whole.txt',header=None)


if file == 'bow':
    data.columns = bow_cols
elif file == 'basic':
    data.columns = bow_cols + bsc
elif file == 'polarity':
    data.columns = bow_cols + plr
elif file == 'graph':
    data.columns = bow_cols + grp
else:
    data.columns = bow_cols + pg

knn = {

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    }
}

svm = {

    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25., 45., 60., 75., 85., 100.]
    } 
}

dt = {

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    }
}


nb = {

    'sklearn.naive_bayes.GaussianNB': {
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 5., 10., 15., 20., 25., 45., 60., 75., 85., 100.],
        'fit_prior': [True, False]
    }  
}

rf = {
        
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    }
}

if limit == 'knn':
    tpot_conf = knn
elif limit == 'svm':
    tpot_conf = svm
elif limit == 'dt':
    tpot_conf = dt
elif limit == 'nb':
    tpot_conf = nb
else:
    tpot_conf = rf

data_shuffle = data
if shuf == 'y':
    data_shuffle = data.iloc[np.random.permutation(len(data))]
data_features = data_shuffle.reset_index(drop=True)
data_class = data_features['class'].values

tpot = TPOTClassifier(generations=400,verbosity=2,cv=10,scoring=opt,template='Classifier',config_dict=tpot_conf)
tpot.fit(data_shuffle.drop('class',axis=1).values, data_class)
print(tpot.score(data_shuffle.drop('class',axis=1).values, data_class))

filename = './exported/tpot-pipeline-' + file + '-' + limit +'.py'
tpot.export(filename)

print("file exported to:", filename)


