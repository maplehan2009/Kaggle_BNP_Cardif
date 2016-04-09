# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:45:08 2016

@author: jingtao
"""
	
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import svm
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectFromModel

def data_pre_processing(m):
    m = m.fillna(0)
    m_numeric = m[m.describe().columns.values] 
    return m_numeric

def feature_selection(m, t):
    y_train = m.values[:, 1]
    X_train = m.values[:, 2:]
    X_test = t.values[:, 1:]
    
    idx = SelectKBest(f_classif, k=5).fit(X_train, y_train).get_support()
    X_train_new = X_train[:, idx]
    X_test_new = X_test[:, idx]
    return X_train_new, y_train, X_test_new

def decomposition_PCA(m, t):
    y_train = m.values[:, 1]
    X_train = m.values[:, 2:]	
    X_test = t.values[:, 1:]
    
    pca = PCA(n_components=10)
    X_train_new = pca.fit_transform(X_train)
    X_test_new = pca.fit_transform(X_test)
    
    return X_train_new, y_train, X_test_new

def learning_algo_svm(X, y):
    clf = svm.SVC(probability=True).fit(X_train, y_train)
    return clf
    
def learning_algo_logit(X, y):
    clf = LogisticRegression().fit(X, y)
    return clf


if __name__=='__main__':
    m = pd.read_csv('data/train.csv')
    t = pd.read_csv('data/test.csv')
    m = data_pre_processing(m)
    t = data_pre_processing(t)
    
    #X_train, y_train, X_test = decomposition_PCA(m, t)
    X_train, y_train, X_test = feature_selection(m, t)
    clf = learning_algo_logit(X_train, y_train)
    #clf = learning_algo_svm(X_train, y_train)
    y_test = clf.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(y_test, index = t.ID, columns=['PredictedProb'])
    submission.to_csv('data/my_submission.csv')
    
