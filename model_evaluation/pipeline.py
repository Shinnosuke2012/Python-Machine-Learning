#Pipelineを使うことで任意の変換ステップを用いて予測ができるようになる。
#例えば、データを標準化してPCAで次元削減したのちに、ロジスティック回帰で分類問題を解くことをPipelineを用いることで変換をスムーズにする

#Breast Cancer Wisconsinデータセットを用いて分析を行う
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#データセットの読み込み
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)

X = df.loc[:,2:].values
y = df.loc[:,1].values

#ラベルエンコーダーでBとMを0 1 の配列に変換する
le = LabelEncoder()
y = le.fit_transform(y)

#データセットの分割
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#Pipelineで変換器と推定器を結合する
pipeline_lr = Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=1))])

pipeline_lr.fit(X_train,y_train)

#精度を計算する
print('Test Accuracy: %.3f'% pipeline_lr.score(X_test,y_test))