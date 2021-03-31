import numpy as np

class Perceptron():
    # eta: 学習率 0から1までのfloat型変数
    # n_iter: 繰り返し回数
    # random_state: 重みを初期化するための乱数seed
    
    # w_: 重みの配列
    # errors_: 各エポックでの誤分類の数
    
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter): 
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta * (target - self.predict(xi)) #重みの更新を行う
                self.w_[1:] += update * xi
                self.w_[0] += update #w_0はx_iは作用しない
                # updateの値が0でない場合には誤分類としてカウントする
                errors += int(update!=0)
            #反復回数ごとの誤差を更新する
            self.errors_.append(errors)
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def predict(self,X):
        return np.where(self.net_input(X) >=0.0, 1, -1)

import numpy as np
import pandas as pd

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

#エポックに対してのご認識の数をプロットする
import matplotlib.pyplot as plt
ppn = Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of errors')
plt.show()

from matplotlib.colors import ListedColormap

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


plot_decision_regions(X,y,classifier=ppn)

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.show()