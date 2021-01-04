import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
features = pd.read_csv('../../elliptic_bitcoin_dataset/full_data.csv',header=None, dtype='float64')
classes = pd.read_csv('../../elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
feature = [str(i) for i in range(170)]
features.columns = ["txId","time_step"] + feature
features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')


for i in range(170):
    fig, axes = plt.subplots(2, 4)
    fig.suptitle(f'Feature {i}')
    features1 = features[features['class'] == '1']
    group = features1[f'{i}'].groupby(features1['time_step'])
    group.mean().plot(ax=axes[0, 0], figsize=(20,10),color='green')
    axes[0, 0].set_title('Licit mean')

    group = features1[f'{i}'].groupby(features1['time_step'])
    group.var().plot(ax=axes[1, 0], figsize=(20,10), color='green')
    axes[1, 0].set_title('Licit variance')

    features2 = features[features['class'] == '2']
    group = features2[f'{i}'].groupby(features2['time_step'])
    group.mean().plot(ax=axes[0, 1], figsize=(20,10), color='red')
    axes[0, 1].set_title('Illicit mean')
    group = features2[f'{i}'].groupby(features2['time_step'])
    group.var().plot(ax=axes[1, 1], figsize=(20,10), color='red')
    axes[1, 1].set_title('Illicit variance')

    featuresu = features[features['class'] == 'unknown']
    group = featuresu[f'{i}'].groupby(featuresu['time_step'])
    group.mean().plot(ax=axes[0, 2], figsize=(20,10))
    axes[0, 2].set_title('Unknown mean')
    group = featuresu[f'{i}'].groupby(featuresu['time_step'])
    group.var().plot(ax=axes[1, 2], figsize=(20,10))
    axes[1, 2].set_title('Unknown variance')


    group = features[f'{i}'].groupby(features['time_step'])
    group.mean().plot(ax=axes[0, 3], figsize=(20,10), color='black')
    axes[0, 3].set_title('Total mean')
    group = features[f'{i}'].groupby(features['time_step'])
    group.var().plot(ax=axes[1, 3], figsize=(20,10), color='black')
    axes[1, 3].set_title('Total variance')

    plt.show()
