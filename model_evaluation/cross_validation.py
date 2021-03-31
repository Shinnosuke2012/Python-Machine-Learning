#k分割交差法を用いて、予測精度を計算する
#Breast Cancer Wisconsinデータセットを用いて分析を行う
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

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

#推定器estimator,トレーニングデータX、予測値y、分割数cv、CPU数n_jobsを指定する
scores = cross_val_score(estimator=pipeline_lr,X=X_train,y=y_train,cv=10,n_jobs=1)

print('CV accuracy score: %s'% scores)