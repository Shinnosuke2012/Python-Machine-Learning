#先ほど行った偏微分のアルゴリズムを用いて、勾配ベクトルを計算してみる
import numpy as np 

def numerical_grad(f,x):
    h = 1e-4 
    grad = np.zeros_like(x) #xと同じ配列を0で初期化する

    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)の計算
        x[idx] = tmp_val + h
        fx1 = f(x)

        #f(x-h)の計算
        x[idx] = tmp_val - h
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2*h)
        #x[idx]をもとにもどす
        x[idx] = tmp_val
    
    return grad 

def function(x):
    return x[0]**2 + x[1]**2 #二変数関数の表し方

print(numerical_grad(function,np.array([3.0,4.0])))