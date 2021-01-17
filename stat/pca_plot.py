#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
features = pd.read_csv('../../elliptic_bitcoin_dataset/elliptic_txs_features.csv',header=None)
classes = pd.read_csv('../../elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
feature = [str(i) for i in range(165)]
features.columns = ["txId","time_step"] + feature
features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')
features['class'] = features['class'].apply(lambda x: '0' if x == "unknown" else x)
data = features[(features['class']=='1') | (features['class']=='2')]
X = data[feature]
Y = data['class']
Y = Y.apply(lambda x: 0 if x == '2' else 1)
pca = PCA(n_components=2).fit(X) 
pcaf = pca.transform(X)

trans = pd.DataFrame()
trans['x'] = pcaf[:,0]
trans['y'] = pcaf[:,1]
trans["illicit"] = Y

plt.figure(figsize=(16,10))
sns.scatterplot(
x="x", y="y",
hue="illicit",
palette=sns.color_palette("hls",2),
data=trans,
legend="full",
alpha=0.3
)

plt.savefig('../image/llicit2D.png')