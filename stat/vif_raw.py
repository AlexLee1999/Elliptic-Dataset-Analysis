#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
features = pd.read_csv('../../elliptic_bitcoin_dataset/elliptic_txs_features.csv',header=None, dtype='float64')
classes = pd.read_csv('../../elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
feature = [str(i) for i in range(165)]
features.columns = ["txId","time_step"] + feature
features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')
features['class'] = features['class'].apply(lambda x: '0' if x == "unknown" else x)
data = features[(features['class']=='1') | (features['class']=='2')]
X = data[feature]
Y = data['class']
Y = Y.apply(lambda x: 0 if x == '2' else 1 )
X = add_constant(X.dropna())
Seri = pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


fi = open('./vif_raw.txt', 'w')
fi.write(f"{Seri.to_string()}")
fi.close()
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
Seri.plot.bar(logy=True)
plt.savefig("../image/vif_raw.png")
