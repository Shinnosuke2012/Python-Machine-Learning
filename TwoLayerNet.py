#誤差逆伝播法を用いた勾配降下法
import sys,os
sys.path.append(os.pardir)
import numpy as np 
from common_function import *
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        #重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size,output_size)

        #レイヤーの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['ReLu'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W2']),self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self,X):
        for layer in layers.values():
            X = layer.forward(X)
        
        return X 
    
    def loss(self,X,t):
        y = self.predict(X)
        return self.lastLayer.forward(y,t)
    
    def accuracy(self,X,t):
        y = self.predict(X)
        y = np.argmax(y,axis=1)
        if t.ndim != 1:
            t = np.argmax(t,axis=1)
        accuracy = np.sum(y==t) / float(X.shape[0])

        return accuracy 
    
    def numerical_gradient(self,X,t):
        loss_W = lambda W: self.loss(X,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return grads 
    
    def gradient(self,X,t):
        #backpropagationを用いた勾配降下法
        #forward
        self.loss(X,t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values()) #レイヤーをリスト化する　Affine1 ReLu Affine2の順
        layers.reverse() #逆伝播は逆から行うため本来のものとは順番が逆になる
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db 

        return grads 