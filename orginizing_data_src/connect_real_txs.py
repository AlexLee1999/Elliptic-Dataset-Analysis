#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage : 
- Input Result file
- Output Json key = id and value = real id
- Change file name if you need
'''
import json
import numpy as np
result_file = '../../elliptic_bitcoin_dataset/Result.csv'
json_file = '../../elliptic_bitcoin_dataset/result.json'
file_features = '../../elliptic_bitcoin_dataset/elliptic_txs_features.csv'
output_csv = "../../elliptic_bitcoin_dataset/full_data.csv"


def read_result_file():
    with open(result_file) as f:
        FH = np.genfromtxt(f, delimiter=',', dtype='str')
    return FH

def read_features_file():
    with open(file_features) as f:
        FH = np.genfromtxt(f, delimiter=',', dtype='str')
    return FH


def output_json(arr):
    dict1 = {arr[i][0] : arr[i][1] for i in range(len(arr))}
    with open(json_file, 'w') as fp:
        json.dump(dict1, fp)
    return dict1

    
if __name__ == "__main__":
    result = read_result_file()
    features = read_features_file()
    a = np.zeros((203769,5))
    features = np.concatenate((features, a), 1)
    result = result[1:]
    
    dic = output_json(result)
    for i in range(len(features)):
        if features[i][0] in dic:
            with open(f'../../txs/{dic[features[i][0]]}.json') as f:
                data = json.load(f)
                if 'block_height' in data:
                    features[i][-1] = data['block_height']
                else:
                    features[i][-1] = np.nan
                if 'weight' in data:
                    features[i][-2] = data['weight']
                else:
                    features[i][-2] = np.nan
                if 'vin_sz' in data:
                    features[i][-3] = data['vin_sz']
                else:
                    features[i][-3] = np.nan
                if 'vout_sz' in data:
                    features[i][-4] = data['vout_sz']
                else:
                    features[i][-4] = np.nan
                if 'size' in data:
                    features[i][-5] = data['size']
                else:
                    features[i][-5] = np.nan
        else:
            features[i][-5] = np.nan
            features[i][-4] = np.nan
            features[i][-3] = np.nan
            features[i][-2] = np.nan
            features[i][-1] = np.nan
    np.savetxt(output_csv, features, delimiter=",",fmt="%s")
