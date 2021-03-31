#コスト関数の勾配降下により重みを更新する

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class LrGD():
    # 属性の説明
    # w_: 適合後の重み
    # cost: コスト関数

    def __init__(self,eta,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state 
    
    def fit(self,X,y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = -y.dot(np.log(output)) - (1-y).dot(np.log(1-output))
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

#データセットの分割
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=1,stratify=y)

#境界面のプロット
lrgd = LrGD(eta=0.05,n_iter=1000,random_state=1)
lrgd.fit(X_train,y_train)

import mlxtend
mlxtend.plot_decision_regions(X_train,y_train,lrgd)
plt.xlabel("Sepal Length (train data)")
plt.ylabel("Petal Length (test data)")
plt.title("Logistic Regression - Gradient Decent")
plt.legend(loc="upper left")
plt.show()