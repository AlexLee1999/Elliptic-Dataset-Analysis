#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
if __name__ == "__main__":
    with open("../elliptic_bitcoin_dataset/clear_data.csv") as f:
        data = np.genfromtxt(f, delimiter=',')

    X = data[:,3:]
    Y = data[:,:1]
    for i in range(len(Y)):
        if Y[i] == 2:
            Y[i] = 0
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0,shuffle=True)

    clf = LogisticRegressionCV(cv=10, max_iter=2000, random_state=0).fit(X_train, Y_train)
    _predict = clf.predict(X_test)
    precision,recall,f1,_ = precision_recall_fscore_support(Y_test,_predict)
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"F1 = {f1}")
    print(classification_report(Y_test,_predict))