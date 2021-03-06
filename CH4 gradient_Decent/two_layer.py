import sys,os
sys.path.append(os.pardir)
from common_function import *
import numpy as np 

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

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self,X,t):
        loss_W = lambda W: self.loss(X,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return grads 

net = TwoLayerNet(input_size=784,hidden_size=100,output_size=10)
X = np.random.rand(100,784)
y = net.predict(X)
t = np.random.rand(100,10)

grads = net.numerical_gradient(X,t)
print(grads['W1'])