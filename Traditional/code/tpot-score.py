#

# For citation, please check:
# https://github.com/teghub/IronyTR

# This code is a helper for scoring the models you get as an output from TPOT [https://epistasislab.github.io/tpot/]. You can add your models manually.
# You may need to install additional libraries for this code to work.
# You may need to create the missing files.

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import SCORERS
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

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


print("features? (bow/basic/polarity/graph/polarity-graph):")
file = input()
print("algo? (svm/knn/nb/tree):")
algo = input()
print("shuffled? (y/n):")
shuf = input()

data = pd.read_csv('../data/features/' + file + '-features-whole.txt',header=None)


# Only best performing pipeline for our data set is shared here.

# bow_k =
bow_n = MultinomialNB(alpha=0.1, fit_prior=True)
# bow_s =
# bow_r =
# bow_t =

# basic_k =
basic_n = MultinomialNB(alpha=1.0, fit_prior=False)
# basic_s =
# basic_r =
# basic_t =

# polarity_k =
polarity_n = MultinomialNB(alpha=0.1, fit_prior=True)
# polarity_s =
# polarity_r = 
# polarity_t = 

# graph_k =
graph_n = MultinomialNB(alpha=1.0, fit_prior=False)
# graph_s = 
# graph_r =
# graph_t = 

# polarity_graph_k =
polarity_graph_n = MultinomialNB(alpha=1.0, fit_prior=False)
# polarity_graph_s =
# polarity_graph_r = 
# polarity_graph_t = 

if file == 'bow':
    data.columns = bow_cols
    if algo == 'knn':
        pipeline = bow_k
    elif algo == 'nb':
        pipeline = bow_n
    elif algo == 'svm':
        pipeline = bow_s
    elif algo == 'rf':
        pipeline = bow_r
    else:
        pipeline = bow_t

elif file == 'basic':
    data.columns = bow_cols + bsc
    if algo == 'knn':
        pipeline = basic_k
    elif algo == 'nb':
        pipeline = basic_n
    elif algo == 'svm':
        pipeline = basic_s
    elif algo == 'rf':
        pipeline = basic_r
    else:
        pipeline = basic_t

elif file == 'polarity':
    data.columns = bow_cols + plr
    if algo == 'knn':
        pipeline = polarity_k
    elif algo == 'nb':
        pipeline = polarity_n
    elif algo == 'svm':
        pipeline = polarity_s
    elif algo == 'rf':
        pipeline = polarity_r
    else:
        pipeline = polarity_t

elif file == 'graph':
    data.columns = bow_cols + grp
    if algo == 'knn':
        pipeline = graph_k
    elif algo == 'nb':
        pipeline = graph_n
    elif algo == 'svm':
        pipeline = graph_s
    elif algo == 'rf':
        pipeline = graph_r
    else:
        pipeline = graph_t

else: #polarity-graph
    data.columns = bow_cols + pg
    if algo == 'knn':
        pipeline = polarity_graph_k
    elif algo == 'nb':
        pipeline = polarity_graph_n
    elif algo == 'svm':
        pipeline = polarity_graph_s
    elif algo == 'rf':
        pipeline = polarity_graph_r
    else:
        pipeline = polarity_graph_t

shuffled = data
if shuf == 'y':
    shuffled = data.iloc[np.random.permutation(len(data))]

a = 0
p = 0
r = 0
f = 0

for fold in range(10):
    test = shuffled.iloc[(fold*60):((fold*60)+60)]
    testing = test.reset_index(drop=True)
    test_class = testing['class'].values
    pre = shuffled.iloc[0:(fold*60)]
    post = shuffled.iloc[((fold*60)+60):600]
    train = pd.concat([pre,post],axis=0)
    training = train.reset_index(drop=True)
    train_class = training['class'].values
    pipeline.fit(train.drop('class',axis=1).values, train_class)
    aac = SCORERS['accuracy'](pipeline, test.drop('class',axis=1).values, test_class)
    ppre = SCORERS['precision'](pipeline, test.drop('class',axis=1).values, test_class)
    rrec = SCORERS['recall'](pipeline, test.drop('class',axis=1).values, test_class)
    ff1 = SCORERS['f1'](pipeline, test.drop('class',axis=1).values, test_class)
    # to see fold scores
    # print(aac, ppre, rrec, ff1)
    a += aac
    p += ppre
    r += rrec
    f += ff1

print("accuracy:", a/10)
print("precision:", p/10)
print("recall:", r/10)
print("f1:", f/10)