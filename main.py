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

def data_pre_processing(m):
    m = m.fillna(0)
    m_numeric = m[m.describe().columns.values]
    #m_numeric = m_numeric.dropna() 
    return m_numeric

def feature_selection(m):
    y = m.values[:, 1]
    X = m.values[:, 2:]
    X_new = SelectKBest(f_classif, k=25).fit_transform(X, y)
    return X_new, y

def decomposition_PCA(m):
    y = m.values[:, 1]
    X = m.values[:, 2:]	
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(X) 
    return X_new, y

def learning_algo_svm(X, y):
    clf = svm.SVC().fit(X_train, y_train)
    return clf
    
def learning_algo_logit(X, y):
    clf = LogisticRegression().fit(X, y)
    return clf
    


if __name__=='__main__':
    m = pd.read_csv('data/train.csv')
    m = data_pre_processing(m)
    X, y = decomposition_PCA(m)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split( X, y, test_size=0.3, random_state=0)
    clf = learning_algo_logit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print score
