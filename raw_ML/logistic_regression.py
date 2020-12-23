#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
from shared import prepare_data


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = prepare_data()
    clf = LogisticRegression().fit(X_train, Y_train)
    _predict = clf.predict(X_test)
    cm = confusion_matrix(Y_test, _predict, labels=clf.classes_)
    precision, recall, f1, _ = precision_recall_fscore_support(Y_test, _predict)
    print(f"precision = {precision[1]}")
    print(f"recall = {recall[1]}")
    print(f"F1 = {f1[1]}")
    print(classification_report(Y_test, _predict))
    print(cm)