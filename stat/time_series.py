import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
features = pd.read_csv('../elliptic_bitcoin_dataset/elliptic_txs_features.csv',header=None, dtype='float64')
classes = pd.read_csv('../elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
feature = [str(i) for i in range(165)]
features.columns = ["txId","time_step"] + feature

for i in range(165):
    group = features[f'{i}'].groupby(features['time_step'])
    group.mean().plot()
    plt.title(f'feature {i}')
    plt.show()

for i in range(165):
    group = features[f'{i}'].groupby(features['time_step'])
    group.var().plot()
    plt.title(f'feature {i}')
    plt.show()