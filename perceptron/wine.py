from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

#Wineデータセットの分析を行う
wine = datasets.load_wine()
feature_names = wine.feature_names

X = wine.data #特徴量
y = wine.target #正解ラベル

#全体の30%をテストデータとして用いる
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#分類をするためには、データを標準化しておくことが望ましい
X_train_std = (X_train - X_train.mean()) /X_train.std()
X_test_std = (X_test - X_test.mean()) /X_test.std()

#正規化パラメータの度合いを変えて、重み係数の変動を調べる
fig = plt.figure(figsize=(12,8))
ax = plt.subplot(111)

colors = ['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']

weights,params = [],[]

for c in [10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),1,10**1,10**2,10**3,10**4,10**5]:
    lr = LogisticRegression(penalty='l2',C=c,random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(c)

weights = np.array(weights)

for column,color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,column],label=feature_names[column],color=color)

#y=0に線を設ける
plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5),10**5])
plt.xlabel("C")
plt.ylabel("weight coef")
plt.xscale('log')
plt.legend(loc="upper left")
ax.legend(loc="upper center",bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)
plt.show()