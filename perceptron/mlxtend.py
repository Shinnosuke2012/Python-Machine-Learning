#決定領域のプロット
def plot_decision_regions(X, y, classifier, resolution=0.01):
    
  from matplotlib.colors import ListedColormap
  import matplotlib.pyplot as plt 
  import numpy as np 

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
  plt.figure(figsize=(10,6))
  plt.contourf(xx1, xx2, z, alpha=0.1)
  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())

  for i, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[i], marker=markers[i], label=cl,edgecolor="k")
