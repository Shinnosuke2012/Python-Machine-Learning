#勾配降下法を用いた単層ニューラルネットワークの構築を行う
import sys,os 
sys.path.append(os.pardir)
import numpy as np 
from common_function import softmax,cross_entropy,numerical_gradient,sigmoid

class simpleNet():
    def __init__(self):
        self.W = np.random.randn(2,3) #2*3のガウス分布で初期化する
    
    def predict(self,X):
        self.a = np.dot(X,self.W)
        self.z = sigmoid(self.a)
        self.y = softmax(self.z)
        return self.y
    
    def loss_cal(self,X,t):
        self.y = self.predict(X)
        self.loss = cross_entropy(self.y,t)

        return self.loss

X = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss_cal(X, t)
dW = numerical_gradient(f, net.W)

print(dW)