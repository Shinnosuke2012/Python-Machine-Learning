import numpy as np 

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def cross_entropy(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    #logの部分にlog(0)となって発散しないように微小値を足しておく
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


#ReLuレイヤーの生成
class ReLu():
    def __init__(self):
        self.mask = None
    
    def forward(self,X):
        self.mask = (X<=0) #boolean
        out = X.copy()
        out[self.mask] = 0

        return out 
    
    def backward(self,dout):
        dout[self.mask] = 0
        dX = dout

        return dX 

#Affineレイヤーの実装を行う
import numpy as np 

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X

        out = np.dot(X, self.W) + self.b

        return out

    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dX

#Softmaxと交差エントロピーを組み合わせたSoftmaxwithlossレイヤーを定義する
class SoftmaxWithLoss():
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self,X,t):
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy(self.y,self.t)

        return self.loss 
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dX = (self.y - self.t) / batch_size #データ1個あたりの誤差が前レイヤーに伝播する

        return dX