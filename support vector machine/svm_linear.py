#SVMの基本
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

#線形SVMのインスタンス生成
svm = SVC(kernel='linear',C=1.0,random_state=0)
svm.fit(X_std,y)

#境界面のプロット
from mlxtend import plot_decision_regions
plot_decision_regions(X_std,y,svm)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend(loc="upper left")
plt.show()