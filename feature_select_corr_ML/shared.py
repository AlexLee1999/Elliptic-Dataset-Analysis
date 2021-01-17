#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


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
    
    cor = []
    for i in range(171):
        x = X[f'{i}'].corr(Y)
        cor.append(x)
    corr = X.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[0]):
            if corr.iloc[i,j] >= 0.99:
                if columns[j] and abs(cor[j]) < abs(cor[i]):
                    if columns[i]:
                        columns[j] = False
                elif columns[i] and abs(cor[j]) > abs(cor[i]):
                    if columns[j]:
                        columns[i] = False
    if num == 0:
        fi = open('./deleted.txt', 'w')
        for i in range(len(columns)):
        
            fi.write(f'{i} : {columns[i]}\n')
        fi.close()
        

    selected_columns = X.columns[columns]
    
    X = X[selected_columns]
    if num == 0:
        cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
        sns.heatmap(X.corr(), cmap=cmap, cbar={'shrink':0.4, 'ticks':[-1, -0.5, 0, 0.5, 1]})
        plt.savefig('../image/corr_feature_select.png')
        plt.close()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=0,shuffle=False)
    return X_train, X_test, Y_train, Y_test
