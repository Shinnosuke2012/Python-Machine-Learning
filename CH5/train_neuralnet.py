import sys,os 
sys.path.append(os.pardir)
import numpy as np 
from mnist import load_mnist
from TwoLayerNet import TwoLayerNet

#最後にbackpropagationを用いたニューラルネットワークの学習を行う

#データの読み込みとネットワークの構築
(X_train,t_train),(X_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

n_iter = 10000
train_size = X_train.shape[0]
batch_size = 100
eta = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size,batch_size,1) #100

for i in range(n_iter):
    batch_mask = np.random.choice(train_size,batch_size)
    X_batch = X_train[batch_mask]
    t_batch = t_train[batch_mask]

    #backpropagationを用いた勾配の学習
    grads = network.gradient(X_batch,t_batch)

    #重みの更新
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= -eta * grads[key]
    
    loss = network.loss(X_batch,t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch==0:
        train_acc = network.accuracy(X_train,t_train)
        test_acc = network.accuracy(X_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)