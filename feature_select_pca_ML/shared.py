#!/usr/bin/env python
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
    features = pd.read_csv('../../elliptic_bitcoin_dataset/full_data.csv',header=None, dtype='float64')
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
    Y = Y.apply(lambda x: 0 if x == '2' else 1 )
    std = StandardScaler()
    X = std.fit_transform(X)
    pca = PCA(n_components = X.shape[1])
    X = pca.fit_transform(X)
    if num == 0:
        cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
        sns.heatmap(pd.DataFrame(X).corr(), cmap=cmap, cbar={'shrink':0.4, 'ticks':[-1, -0.5, 0, 0.5, 1]})
        plt.savefig('../image/corr_pca.png')
        plt.close()
        fi = open('./corr_pca.txt', 'w')
        fi.write(f"{pd.DataFrame(X).corr().to_string()}")
        fi.close()
        X2 = sm.add_constant(X)
        est = sm.OLS(Y, X2)
        est2 = est.fit()
        print("Linear regression")
        print(est2.summary())
        fi = open('./linear_pca_raw.txt', 'w')
        fi.write(f"{est2.summary()}")
        fi.close()
    del_lst = [7, 9, 14, 15, 16, 21, 41, 43, 53, 56, 64, 70, 74, 78, 79, 80, 83, 85, 87, 88, 89, 94, 96, 99, 100, 104, 105, 107, 108, 109, 122, 124, 129, 131, 132, 133, 134, 138, 139, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
    X = np.delete(X, del_lst, axis=1)
    if num == 0:
        X2 = sm.add_constant(X)
        est = sm.OLS(Y, X2)
        est2 = est.fit()
        print("Linear regression")
        print(est2.summary())
        fi = open('./linear_pca_modified.txt', 'w')
        fi.write(f"{est2.summary()}")
        fi.close()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=0,shuffle=False)
    return X_train, X_test, Y_train, Y_test
prepare_data(0)