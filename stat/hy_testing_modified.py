import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
features = pd.read_csv('../../elliptic_bitcoin_dataset/full_data.csv',header=None, dtype='float64')
classes = pd.read_csv('../../elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
feature = [str(i) for i in range(170)]
features.columns = ["txId","time_step"] + feature
features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')
features['class'] = features['class'].apply(lambda x: '0' if x == "unknown" else x)
data = features[(features['class']=='1') | (features['class']=='2')]
X = data[feature]
Y = data['class']
Y = Y.apply(lambda x: 0 if x == '2' else 1 )
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print("Linear regression")
print(est2.summary())

sns.heatmap(X.corr())
plt.show()
