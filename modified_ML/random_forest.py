#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == "__main__":
    features = pd.read_csv('../../elliptic_bitcoin_dataset/full_data.csv',header=None)
    classes = pd.read_csv('../../elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    edges = pd.read_csv('../../elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    tx_features = ["tx_feat_"+str(i) for i in range(2,95)]
    agg_features = ["agg_feat_"+str(i) for i in range(1,75)]
    features.columns = ["txId","time_step"] + tx_features + agg_features
    features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')
    features['class'] = features['class'].apply(lambda x: '0' if x == "unknown" else x)
    data = features[(features['class']=='1') | (features['class']=='2')]
    X = data[tx_features+agg_features]
    Y = data['class']
    Y = Y.apply(lambda x: 0 if x == '2' else 1 )

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=0,shuffle=False)

    rf = RandomForestClassifier(n_estimators=50, max_depth=100,random_state=0).fit(X_train, Y_train)
    _predict = rf.predict(X_test)
    cm = confusion_matrix(Y_test, _predict, labels=rf.classes_)
    precision,recall,f1,_ = precision_recall_fscore_support(Y_test,_predict)
    print(f"precision = {precision[1]}")
    print(f"recall = {recall[1]}")
    print(f"F1 = {f1[1]}")
    print(classification_report(Y_test,_predict))
    print(cm)
    