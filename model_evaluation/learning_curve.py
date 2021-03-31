#学習曲線を用いて、バイアスとバリアンスの評価を行う。

#Breast Cancer Wisconsinデータセットを用いて分析を行う
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt 

#データセットの読み込み
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)

X = df.loc[:,2:].values
y = df.loc[:,1].values

#ラベルエンコーダーでBとMを0 1 の配列に変換する
le = LabelEncoder()
y = le.fit_transform(y)

#データセットの分割
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

#Pipelineで変換器と推定器を結合する
pipeline_lr = Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=1))])

#learning_curve関数で交差検証による正解率を算出
train_sizes,train_scores,test_scores = learning_curve(estimator=pipeline_lr,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)

#k回のイテレーションのスコアの平均値
train_mean = np.mean(train_scores,axis=1)

#k回のイテレーションのスコアの標準偏差
train_std = np.std(train_scores,axis=1)

#k回のイテレーションのスコアの平均値
test_mean = np.mean(test_scores,axis=1)

#k回のイテレーションのスコアの標準偏差
test_std = np.std(test_scores,axis=1)

#トレーニングデータの様子
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean + train_std, train_mean - train_std,alpha=0.15,color='blue')

#テストデータの様子
plt.plot(train_sizes,test_mean,color='green',marker='s',linestyle='--',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean + test_std, test_mean - test_std,alpha=0.15,color='green')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.ylim([0.8,1.0])
plt.show()