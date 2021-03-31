#コスト関数の勾配降下により重みを更新する
#ただし、コスト関数には重みを小さくするためのバイアス（L2正規化)を加える

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class LrGD_reg():
    # 属性の説明
    # w_: 適合後の重み
    # cost: コスト関数

    def __init__(self,eta,C,n_iter=50,random_state=1):
        self.eta = eta
        self.C = C
        self.n_iter = n_iter
        self.random_state = random_state 
    
    def fit(self,X,y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors) + self.eta*(1/self.C)*self.w_[1:] #バイアスを加える
            self.w_[0] += self.eta*errors.sum() + self.eta*(1/self.C)*self.w_[0] #バイアスを加える
            cost = -y.dot(np.log(output)) - (1-y).dot(np.log(1-output)) + (1/(2*self.C))*self.w_.dot(self.w_) #バイアスを加える
            self.cost_.append(cost)
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def activation(self,z):
        #活性化関数にシグモイド関数を用いる
        return 1.0 /(1.0 + np.exp(-np.clip(z,-250,250)))
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, -1)

#データセットの生成
from sklearn import datasets 
iris = datasets.load_iris()
X_raw = iris.data[:,[2,3]]
y_raw = iris.target
X = []
y = []

for i in range(0,len(y_raw)):
    if (y_raw[i]==0) | (y_raw[i]==1):
        X.append(X_raw[i])
        y.append(y_raw[i])

X = np.array(X)
y = np.array(y)

#重みの逆正規化パラメータCによる変化を調べる
weights,params = [],[]
for c in [10**-5,10**-4,10**-3,10**-2,10**-1,1,10**1,10**2,10**3,10**4,10**5]:
    lrgd_reg = LrGD_reg(eta=0.05,C=c,n_iter=100,random_state=1)
    lrgd_reg.fit(X,y)
    weights.append(lrgd_reg.w_[1:])
    params.append(lrgd_reg.C) #スケーリングの問題である

weights = np.array(weights)
params = np.array(params)
# 横軸に逆正規化パラメータ、縦軸に重み係数をプロットする
plt.plot(params,weights[:,0],label="sepal-length")
plt.plot(params,weights[:,1],linestyle='--',label="petal-length")
plt.xlabel("C")
plt.ylabel("weight coefficient")
plt.legend(loc="upper left")
plt.xscale('log')
plt.show()