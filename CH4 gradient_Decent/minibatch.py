import sys,os
sys.path.append(os.pardir) #親ディレクトリーのファイルをインポート
from mnist import load_mnist
from PIL import Image
import pickle
import numpy as np 
from common_function import *

class TwoLayerNet(object):
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        #重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self,X):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']
        
        a1 = np.dot(X,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)
        
        return y 
    
    def loss(self,X,t):
        y = self.predict(X)
        return cross_entropy(y,t)
    
    def accuracy(self,X,t):
        y = self.predict(X)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        
        accuracy = np.sum(y==t) / float(X.shape[0])
    
    def numerical_gradient(self,X,t):
        loss_W = lambda W: self.loss(X,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return grads 

#学習と予測
(X_train,t_train),(X_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

train_loss_list = []

#Hyper parameter 
n_iter = 10
train_size = X_train.shape[0]
batch_size = 100
eta = 0.1

network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

for i in range(n_iter):
    batch_mask = np.random.choice(train_size,batch_size) #10000個のデータから100個を無作為に抽出する
    X_batch = X_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #勾配の計算
    grad = network.numerical_gradient(X_batch,t_batch)
    
    #パラメーターの更新
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= -eta*grad[key]
    
    #学習経過の記録
    loss = network.loss(X_batch,t_batch)
    train_loss_list.append(loss)

print(train_loss_list)