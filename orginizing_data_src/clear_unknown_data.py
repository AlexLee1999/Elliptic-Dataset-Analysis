#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage : 
- Input feature file and class file
- Output feature file with only known nodes
- Change file name if you need
'''
import numpy as np
file_features = '../../elliptic_bitcoin_dataset/full_data.csv'
file_class = '../../elliptic_bitcoin_dataset/elliptic_txs_classes.csv'
clear_data_csv = "../../elliptic_bitcoin_dataset/clear_data.csv"
unknown_data_csv = "../../elliptic_bitcoin_dataset/unknown_data.csv"
def read_features_file():
    with open(file_features) as f:
        FH = np.genfromtxt(f, delimiter=',', dtype='str')
    return FH

def read_file_class():
    with open(file_class) as f:
        FH = np.genfromtxt(f, delimiter=',', dtype='str')
    return FH

def clear_unknown(features, classes):
    del_lst = []
    for i in range(len(features)):
        if classes[i][1] == 'unknown':
            del_lst.append(i)
    for i in range(len(del_lst)):
        features[del_lst[i]][0] = '-1'
    np.savetxt(unknown_data_csv, features[del_lst], delimiter=",",fmt="%s")
    features = np.delete(features, del_lst, axis=0)
    np.savetxt(clear_data_csv, features, delimiter=",",fmt="%s")
    
if __name__ == "__main__":
    results_class = read_file_class()
    results_features = read_features_file()
    results_class = results_class[1::]
    clear_unknown(results_features, results_class)