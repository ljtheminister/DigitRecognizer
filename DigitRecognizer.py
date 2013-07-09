#!/usr/bin/python

import csv
import pandas as pd, numpy as np
# load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
output = pd.load('Classifications')
# create output
'''
output = pd.DataFrame(index=xrange(len(test)))
output['ImageID'] = [i+1 for i in xrange(len(test))]
print 'classifications'
'''
## MULTICLASS CLASSIFIERS
'''
# Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10)
rfc.fit(train.ix[:,1:785], train.ix[:,0])

rf_pred = rfc.predict(test)
output['RF'] = rf_pred

print 'RF'

numpy.savetxt('rf_std10.csv', rf_pred, delimiter=',')
rf_pred = pd.DataFrame(rf_pred)
rf_pred['ImageID'] = rf_pred.index + 1
rf_pred['Label'] = rf_pred.ix[:,0]
rf_pred = rf_pred.ix[:, ['ImageID', 'Label']]
rf_pred.to_csv('rf_std10.csv', header=True, index=False)

# k-NN (k-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(n_neighbors=10)
kNN.fit(train.ix[:,1:785], train.ix[:,0])
kNN_pred = kNN.predict(test)
output['kNN'] = kNN_pred
print 'kNN'
kNN_pred = pd.DataFrame(index=xrange(0,len(test))
kNN_pred = ['ImageID'] = [i+1 for i in xrange(0,len(test))]
kNN_pred['Label'] = kNN.predict(test)
kNN_pred.to_csv('kNN.csv', header=True, index=False)
'''
'''
# Do PCA on Training
from sklearn import decomposition
pca = decomposition.PCA(n_components = 28)
pca.fit(train.ix[:,1:785])
train_pca = pca.transform(train.ix[:, 1:785])
test_pca = pca.transform(test)

print 'PCA'
## ONE-VS-ONE CLASSIFIERS

# Linear Support Vector Classifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
output['LinSVC_1vs1']= OneVsOneClassifier(LinearSVC(random_state=0)).fit(train_pca, train.ix[:,0]).predict(test_pca)

print '1-1 LinSVC'

# SVC - Gaussian kernel
from sklearn.svm import SVC
svc = SVC()
output['SVC'] = svc.fit(train_pca, train.ix[:,0]).predict(test_pca)
print 'SVC'

## ONE-VS-ALL CLASSIFIERS
from sklearn.multiclass import OnevsRestClassifier
'''
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()
output['Logit']= logit.fit(train.ix[:,1:785], train.ix[:,0]).predict(test)
print 'Logistic Regression'
'''
# "one-vs-rest" multi-class classification
lin_svc = LinearSVC()
output['LinSVC_1vsRest']= lin_svc.fit(train_pca, train.ix[:,0]).predict(test_pca)

print 'Linear SVC'
'''
# ENSEMBLE METHOD - Decision Tree, RF, SVM, LinearSVM - majority vote
# create a dataframe with ImageID and all classifications by different algorithms
output.save('Classifications')
df['Label'] = df.ix[:, 1:].apply(lambda x: x.value_counts().idxmax(), axis=1)
output = output.ix[:, ['ImageID', 'Label']]
output.save('Labels')
# Gotta save to CSV

output.to_csv('rf+kNN+logit.csv', sep=',',header=True,index=False)
