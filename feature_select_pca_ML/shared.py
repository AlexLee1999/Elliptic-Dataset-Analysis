#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

def prepare_data(num):
    features = pd.read_csv('../../elliptic_bitcoin_dataset/full_data.csv',header=None)
    classes = pd.read_csv('../../elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    feature = [str(i) for i in range(171)]
    features.columns = ["txId","time_step"] + feature
    features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')
    features['class'] = features['class'].apply(lambda x: '0' if x == "unknown" else x)
    features.dropna(subset=['165'], inplace=True)
    features.dropna(subset=['166'], inplace=True)
    features.dropna(subset=['167'], inplace=True)
    features.dropna(subset=['168'], inplace=True)
    features.dropna(subset=['169'], inplace=True)
    features.dropna(subset=['170'], inplace=True)
    data = features[(features['class']=='1') | (features['class']=='2')]
    X = data[feature]
    Y = data['class']
    Y = Y.apply(lambda x: 0 if x == '2' else 1)
    std = StandardScaler()
    X = std.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=0,shuffle=False)
    pca = PCA(n_components = X.shape[1] - 1)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    del_lst = [30, 37, 44, 49, 50, 51, 59, 60, 65, 66, 72, 75, 76, 77, 78, 85, 86, 87, 89, 90, 91, 92, 93, 96, 97, 99, 100, 101, 102, 104, 106, 107, 110, 112, 115, 119, 120, 122, 124, 125, 127, 128, 129, 131, 132, 134, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169]
    X_train = np.delete(X_train, del_lst, axis=1)
    X_test = np.delete(X_test, del_lst, axis=1)
    return X_train, X_test, Y_train, Y_test