import pandas as pd
import data_explorer as d
import randomForest as rf
import xgBoost as xg
import naive_bayes as nb
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import NMF
from scipy import stats
import random
import numpy as np

# Read in training set
training_data = pd.read_csv('mediaeval-2015-trainingset.txt', delimiter='\t')
# Read in test set
testing_data = pd.read_csv('mediaeval-2015-testset.txt', delimiter='\t')

# Return cleaned data set
train = d.setup_train_data(training_data)
test = d.setup_data(testing_data)

# Prepare for classifier
train_target = train['label']
test_target = test['label']
train = train.drop(['label'], axis=1)
test = test.drop(['label'], axis=1)

print(nb.classify(train, train_target, test, test_target))


""" No longer needed
kbest = SelectKBest(f_classif, k=3)
train = kbest.fit_transform(train, train_target)
test = kbest.transform(test)

Resample data set to solve imbalance
resampler = RandomOverSampler()
train, train_target = resampler.fit_resample(train, train_target) """