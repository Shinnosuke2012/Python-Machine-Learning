# 確率的勾配降下法を用いた線形分離を行う
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class AdalineSGD():
    def __init__(self,eta,n_iter,shuffle=True,random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False #重みの初期化フラグはFalseに設定
        self.shuffle = shuffle
        self.random_state = random_state 
    
    def fit(self,X,y):
        #トレーニングデータに適合
        self._initialize_weights(X.shape[1]) #重みベクトルの生成
        self.cost_ = [] #コスト関数の定義

        #トレーニング回数分だけデータを反復する
        for i in range(self.n_iter):
            #トレーニングデータをシャフルする
            if self.shuffle:
                X,y = self._shuffle(X,y)
            #各試行回数分のコスト関数を格納するリストを生成
            cost = []
            #各サンプルに対する計算
            for xi, target in zip(X,y):
                #特徴量と目的変数を用いた重みの更新とコストの計算
                cost.append(self._update_weights(xi,target))
            #各サンプルにおいて平均コストを計算する
            avg_cost = sum(cost)/len(y)
            #平均コストを格納する
            self.cost_.append(avg_cost)
        return self 

    def partial_fit(self,X,y):
        #重みを初期化することなくトレーニングデータに適合させる
        #初期化されていないときは初期化を実行
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        #目的変数の要素数が2以上の時は各サンプルの特徴量xiとtargetを用いて重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi,target)
        
        #目的変数のyの要素数が1の場合は
        #サンプル全体の特徴量Xとyで重みを更新
        else:
            self._update_weights(X,y)
        return self 


    def _shuffle(self,X,y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self,m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0,scale=0.01,size=1+m)
        self.w_initialized = True
    
    def _update_weights(self,xi,target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta*xi.dot(error)
        self.w_[0] += self.eta*error 
        cost = 0.5*error**2
        return cost 
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def activation(self,X):
        return X
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    

#データセットの読み込み
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


#決定領域のプロット
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


#データの正規化
X_std = np.copy(X)
X_std[:,0] = (X_std[:,0] - X_std[:,0].mean()) /X_std[:,0].std()
X_std[:,1] = (X_std[:,1] - X_std[:,1].mean()) /X_std[:,1].std()


adaSGD = AdalineSGD(eta=0.01,n_iter=15,random_state=1)
adaSGD.fit(X_std,y)
plot_decision_regions(X_std,y,adaSGD)
plt.title("Adaline - Stochastic Gradient Decent")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(loc="upper left")
plt.show()

#コスト関数の収束具合を確かめる
plt.plot(range(1,len(adaSGD.cost_)+1),adaSGD.cost_,marker='o')
plt.xlabel("Epoches")
plt.ylabel("Cost function")
plt.title("Epoches - cost function")
plt.show()

