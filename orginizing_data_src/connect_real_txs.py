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

def read_result_file():
    with open(result_file) as f:
        FH = np.genfromtxt(f, delimiter=',', dtype='str')
    return FH


def output_json(arr):
    dict1 = {arr[i][0] : arr[i][1] for i in range(len(arr))}
    with open(json_file, 'w') as fp:
        json.dump(dict1, fp)
    return dict1

    
if __name__ == "__main__":
    result = read_result_file()
    result = result[1:]
    dic = output_json(result)