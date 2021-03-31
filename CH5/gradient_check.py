#数値微分で求めた勾配と誤差逆伝播法で求めた勾配が限りなく近いことを確認する
import sys,os
sys.path.append(os.pardir)
import numpy as np 
from mnist import load_mnist
from TwoLayerNet import TwoLayerNet

#データの読み込み
(X_train,t_train),(X_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

X_batch = X_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(X_batch,t_batch)
grad_backprop = network.gradient(X_batch,t_batch)

#各重みの絶対誤差の平均を計算する
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
    