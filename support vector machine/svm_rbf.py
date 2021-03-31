#カーネルトリックを使った分類
from sklearn import datasets 
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt 

#データセットの生成
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

#Xを正規化しておく
X_std = (X - X.mean()) /X.std()

#ハイパーパラメータによって決定領域がどのように変わるかを見る
param = [0.10,1.0,10.0,100.0]
label = ['0.1','1.0','10.0','100.0']
for i in range(0,len(param)):
    #線形SVMのインスタンス生成
    svm = SVC(kernel='rbf',C=1.0,gamma=param[i],random_state=0)
    svm.fit(X_std,y)

    #境界面のプロット
    from mlxtend import plot_decision_regions
    plot_decision_regions(X_std,y,svm)
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    plt.legend(loc="upper left")
    plt.title("Gausian Kernel gamma= " + label[i])
    plt.show()