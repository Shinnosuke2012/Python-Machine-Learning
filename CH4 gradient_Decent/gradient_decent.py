#ニューラルネットワークの重みの更新は勾配降下法を用いて行われることが多い。
#ここでは、任意の関数において勾配降下法を用いることで、最小値を求める

import numpy as np 

def numerical_grad(f,x):
    h = 1e-4 
    grad = np.zeros_like(x) #xと同じ配列を0で初期化する

    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)の計算
        x[idx] = float(tmp_val) + h
        fx1 = f(x)

        #f(x-h)の計算
        x[idx] = tmp_val - h
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2*h)
        #x[idx]をもとにもどす
        x[idx] = tmp_val
    
    return grad 

def gradient_decent(f,init_x,eta,n_iter=100):
    x = init_x #xを初期化
    
    for i in range(n_iter):
        grad = numerical_grad(f,x)
        x -= eta*grad 
    
    return x 

def function(x):
    return x[0]**2 + x[1]**2 #二変数関数の表し方

init_x = np.array([-3.0,4.0])
min = gradient_decent(function,init_x,eta=0.1,n_iter=100)
print(min)