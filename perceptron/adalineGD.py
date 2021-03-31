#コスト関数の勾配降下により重みを更新する

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class AdalineGD():
    # 属性の説明
    # w_: 適合後の重み
    # cost: コスト関数

    def __init__(self,eta,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state 
    
    def fit(self,X,y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum() /2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def activation(self,X):
        return X
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# machine-learning-databasesからデータを取得
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
'''
irisデータセット(150, 4)
- がく片の長さ(cm)
- がく片の太さ(cm)
- 花びらの長さ(cm)
- 花びらの太さ(cm)
- タグ（Iris-Setosa, Iris-Versicolour, Iris-Virginica)
  0:50が "Iris-Setosa", 50:100が "Iris-Versicolur", 100:150が "Iris-Virginca"
'''
# がく片の太さ(1)と花びらの太さ(3)を取得
# shape : (100, 2)
X = df.iloc[:100, [1, 3]].values
# タグを取得
# shape : (100, 1)
y = df.iloc[:100, 4].values
# クラスラベルを変換（Iris-Setoraを-1、Iris-Versicolourを１）
y = np.where(y == 'Iris-setosa', -1, 1)


#エポックによってコスト関数の収束具合を確かめる
fig,ax = plt.subplots(1,2,figsize=(10,4))

ada1 = AdalineGD(0.01,50,1)
ada1.fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_))
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(cost_function)')
ax[0].set_title('Adaline - learning_rate=0.01')

ada2 = AdalineGD(0.0001,50,1)
ada2.fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('cost_function')
ax[1].set_title('Adaline - learning_rate=0.0001')

plt.show()

#コスト関数の収束を早めるためにデータの標準化を行う。
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) /X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) /X[:,1].std()

#Irisデータを用いて学習を進める
ada = AdalineGD(eta=0.01,n_iter=50,random_state=1)
ada.fit(X_std,y)

#境界面をプロットする関数の記述
def plot_decision_regions(X, y, classifier, resolution=0.01):
  # マーカーと色の準備
  # 2種類(1か-1)で十分
  markers = ('s', 'x')
  colors = ('red', 'blue')
  mp = ListedColormap(colors[:len(np.unique(y))])
  # 「がく片の太さ」の最大値と最小値を取得。
  # 描画に余裕を持たせるために+1と-1
  x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  # 「花びらの太さ」の最大値と最小値を取得
  # 描画に余裕を持たせるために+1と-1
  x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  # 格子点の座標をresolution(0.01)ごとに取得
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
  # パーセプトロンの分類器を使って、zにデータを格納
  z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  z = z.reshape(xx1.shape)
  # 格子点とデータをもとに線をプロット
  plt.contourf(xx1, xx2, z, alpha=0.1)
  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())

  for i, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[i], marker=markers[i], label=cl,edgecolor="k")

plot_decision_regions(X_std,y,classifier=ada,resolution=0.01)
plt.title('Adaline - Gradient decent')
plt.xlabel("Sepal length(standized")
plt.ylabel("Petal length(standized")
plt.show()

#コスト関数の表示
plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='x')
plt.xlabel("Epochs")
plt.ylabel("Cost function")
plt.title("Epochs - cost function")
plt.show()