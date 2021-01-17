#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    return X_train, X_test, Y_train, Y_test