#入力層784個に対して、出力層10個のニューラルネットワークを構築する

import sys,os
sys.path.append(os.pardir) #親ディレクトリーのファイルをインポート
from mnist import load_mnist
from PIL import Image
import pickle
import numpy as np 

def get_data():
    (X_train,t_train),(X_test,t_test) = load_mnist(flatten=True,normalize=False)
    return X_test,t_test 

#学習済みの重みを読み込む
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    
    return network 

def sigmoid(X):
    return 1/(1 + np.exp(-X))

def soft_max(X):
    C = np.max(X)
    exp_a = np.exp(X - C)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y 

def predict(network,X):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(X,W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2,W3) + b3
    y = soft_max(a3)

    return y 

#分類が上手くいっているかを評価する
X_test,t_test = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(X_test)):
    y = predict(network,X_test[i]) #784個の確率が格納してある配列を生成する
    p = np.argmax(y) #一番確率の高いインデックス番号を取得する

    if p==t_test[i]: 
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt)/len(X_test)))