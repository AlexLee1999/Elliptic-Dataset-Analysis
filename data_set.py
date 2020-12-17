#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv
file = '../elliptic_bitcoin_dataset/elliptic_txs_features.csv'
file_class = '../elliptic_bitcoin_dataset/elliptic_txs_classes.csv'
def read_file():
    reader = csv.reader(open(file, "r"), delimiter=",")
    x = list(reader)
    results = np.array(x).astype("float")
    return results

def read_file_class():
    with open(file_class) as f:
        FH = np.genfromtxt(f, delimiter=',', dtype='str')
    return FH

def clear_unknown(features, classes):
    del_lst = []
    count = 0
    for i in range(1, len(features)):
        if classes[i][1] == 'unknown':
            del_lst.append(i-1)
        else:
            count += 1
    features = np.delete(features, del_lst, axis=0)
    np.savetxt("clear.csv", features, delimiter=",")
    print(count)
    
if __name__ == "__main__":
    results_class = read_file_class()
    results_features = read_file()
    
    clear_unknown(results_features, results_class)