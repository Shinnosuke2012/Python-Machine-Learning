#wineデータに対して主成分分析を行う
import numpy as np 
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 

class PCA():
    def __init__(self):
        pass

    def fit(self,X,y):
        #データを分割する
        self.X_std = 10* ((X - X.mean()) /X.std())

        #共分散行列を生成する
        self.cov_mat = np.cov(self.X_std.T)
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(self.cov_mat)

        #固有値の大きいものに付随する固有ベクトルの成分と各データのベクトルのスカラー積が第一主成分と続いていく
        self.eigen_pairs = [(self.eigen_vals[i],self.eigen_vecs[i]) for i in range(len(self.eigen_vals))]
        self.eigen_pairs.sort(reverse=True)

        #主成分の個数を指定した後、固有値をその個数分つなげた配列を生成する
        self.w_ = np.hstack((self.eigen_pairs[0][1][:,np.newaxis],self.eigen_pairs[1][1][:,np.newaxis]))

        #重みw_とデータXの内積が主成分である。xT w_を計算すると列成分に主成分が並ぶ
        self.X_train_pca = self.X_std.dot(self.w_)
    
    def fit_transform(self,X,y):
        return self.X_train_pca
    
    def tot(self,X,y):
        self.tot = sum(self.eigen_vals)

        #分散説明率を計算
        self.var_exp = [(i / self.tot) for i in sorted(self.eigen_vals,reverse=True)]
        self.cum_var_exp = np.cumsum(self.var_exp)

        #分散説明率のグラフ作成
        plt.bar(range(1,14),self.var_exp,alpha=0.5,align="center",label="individual explained variances")

        #累積分散説明率のグラフを作成
        plt.step(range(1,14),self.cum_var_exp,where="mid",label="cumulative explained variances")

        plt.xlabel("Princial composition")
        plt.ylabel("Explained varience ratio")
        plt.legend(loc="best")
        plt.show()
    
    def plot(self,X,y):
        colors = ['r','b','g']
        markers = ['s','x','o']

        for l,c,m in zip(np.unique(y),colors,markers):
            plt.scatter(self.X_train_pca[y==l,0],self.X_train_pca[y==l,1],c=c,label=l,marker=m)
        
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(loc='lower left')
        plt.show()


#Wineデータセットの分析を行う
wine = datasets.load_wine()
feature_names = wine.feature_names

X = wine.data #特徴量
y = wine.target #正解ラベル
y = y + 1

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)

#インスタンスの生成
pca = PCA()
pca.fit(X_train,y_train)
pca.tot(X_train,y_train)
pca.plot(X_train,y_train)

X_train_pca = pca.fit_transform(X_train,y_train)

#続いて、次元削減したデータを線形分離する
from mlxtend import plot_decision_regions

lrGD = LogisticRegression()
lrGD.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca,y_train,classifier=lrGD)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc="lower left")
plt.show()