#Kernel PCAを実装する
from scipy.spatial.distance import pdist,squareform
from scipy import exp 
from scipy.linalg import eigh 
import numpy as np 
from sklearn.datasets import make_moons 
import matplotlib.pyplot as plt 

class Kernel_PCA():
    def __init__(self,n_components,gamma):
        self.n_components = n_components
        self.gamma = gamma

    def fit(self,X,y):
        #M*N次元のデータセットでペアごとの平方ユークリッド距離を計算
        self.sq_dists = pdist(X,metric='sqeuclidean')

        #ペアごとの値を行列に格納する
        self.mat_sq_dists = squareform(self.sq_dists)

        #ガウシアンカーネル行列を計算
        self.K = exp(-self.gamma*self.mat_sq_dists)

        #カーネル行列を中心化
        N = self.K.shape[0]
        self.one_n = np.ones((N,N)) /N
        self.K = self.K - self.one_n.dot(self.K) - self.K.dot(self.one_n) + self.one_n.dot(self.K).dot(self.one_n)

        #中心化されたカーネル行列から固有値ならびに固有ベクトルを抽出
        self.eigen_vals,self.eigen_vecs = eigh(self.K)

        #上位k個の固有ベクトルを抽出
        self.X_kpca = np.column_stack((self.eigen_vecs[:,-i] for i in range(1,self.n_components + 1)))
    
    def fit_transform(self,X,y):
        return self.X_kpca

#データセットの生成
X,y = make_moons(n_samples=100,random_state=123)
kpca = Kernel_PCA(n_components=2,gamma=15)
kpca.fit(X,y)
X_kpca = kpca.fit_transform(X,y)

fig,ax = plt.subplots(1,2,figsize=(7,3))

ax[0].scatter(X_kpca[y==0,0],X_kpca[y==0,1],c='red',marker='o',edgecolor='k',label='0')
ax[0].scatter(X_kpca[y==1,0],X_kpca[y==1,1],c='blue',marker='o',edgecolor='k',label='1')
ax[1].scatter(X_kpca[y==0,0],np.zeros((50,1))+0.02,c='red',marker='o',edgecolor='k',label='0')
ax[1].scatter(X_kpca[y==1,0],np.zeros((50,1))-0.02,c='blue',marker='o',edgecolor='k',label='1')
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_xlabel('PC1')
ax[1].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
plt.show()

#処理したデータが線形分離可能かを確かめてみる
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2')
lr.fit(X_kpca,y)

from mlxtend import plot_decision_regions

plot_decision_regions(X_kpca,y,classifier=lr)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Linear seperation - Kernel PCA")
plt.legend(loc="upper right")
plt.show()