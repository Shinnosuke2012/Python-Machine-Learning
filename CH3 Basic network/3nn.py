#3層ニューラルネットワークを実装する
#だだし、重みはこちらから与えるものとする

import numpy as np 
import matplotlib.pyplot as plt 

class NN_3(object):
    def __init__(self):
        pass

    def fit(self,X):
        #一層目の変換を行う
        self.W_1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
        self.B_1 = np.array([0.1,0.2,0.3])
        self.A_1 = np.dot(X,self.W_1) + self.B_1
        #シグモイド関数を用いる
        self.Z_1 = self.sigmoid(self.A_1)

        #二層目の変換を行う
        self.W_2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
        self.B_2 = np.array([0.1,0.2])
        self.A_2 = np.dot(self.Z_1,self.W_2) + self.B_2 
        #同様にシグモイド関数を用いる
        self.Z_2 = self.sigmoid(self.A_2)

        #二層目から出力層への信号伝達
        self.W_3 = np.array([[0.1,0.3],[0.2,0.4]])
        self.B_3 = np.array([0.1,0.2])
        self.A_3 = np.dot(self.Z_2,self.W_3) + self.B_3
        #活性化関数はソフトマックスを用いる
        self.y = self.soft_max(self.A_3)
    
    def sigmoid(self,X):
        return  1/ (1 + np.exp(-(X)))
    
    def soft_max(self,X):
        self.C = np.max(X)
        self.exp_A_3 = np.exp(X - self.C)
        self.sum_exp_A_3 = np.sum(self.exp_A_3)
        return self.exp_A_3 / self.sum_exp_A_3

    
    def fit_transform(self,X):
        print('Z1 : ',self.Z_1)
        print('Z2 : ',self.Z_2)
        print('y : ',self.y)

X = np.array([1.0,0.5])

nn_3 = NN_3()
nn_3.fit(X)
nn_3.fit_transform(X)