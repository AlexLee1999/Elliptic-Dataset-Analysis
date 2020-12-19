#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
if __name__ == "__main__":
    with open("../elliptic_bitcoin_dataset/clear_data.csv") as f:
        data = np.genfromtxt(f, delimiter=',')
    with open("../elliptic_bitcoin_dataset/unknown_data.csv") as f:
        unknown_data = np.genfromtxt(f, delimiter=',')
    X_unknown = unknown_data[:,3:]
    X = data[:,3:]
    Y = data[:,:1]
    for i in range(len(Y)):
        if Y[i] == 2:
            Y[i] = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=0,shuffle=True)

    rf = MLPClassifier(hidden_layer_sizes=(50, ),random_state=0, max_iter=300).fit(X_train, Y_train)
    _predict_unknown = rf.predict(X_unknown)
    Y_train = np.concatenate((Y_train, np.array([_predict_unknown]).T), axis=0)
    X_train = np.concatenate((X_train, X_unknown), axis=0)
    rf = MLPClassifier(hidden_layer_sizes=(50, ),random_state=0, max_iter=300).fit(X_train, Y_train)

    _predict = rf.predict(X_test)
    cm = confusion_matrix(Y_test, _predict, labels=rf.classes_)
    precision,recall,f1,_ = precision_recall_fscore_support(Y_test,_predict)
    print(f"precision = {precision[1]}")
    print(f"recall = {recall[1]}")
    print(f"F1 = {f1[1]}")
    print(classification_report(Y_test,_predict))
    print(cm)

    