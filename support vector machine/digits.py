import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import svm,datasets
from sklearn.metrics import accuracy_score,classification_report

digits = datasets.load_digits()
cnt = 1

#データの可視化
for i in range(0,10):
    plt.subplot(2,5,cnt)
    cnt = cnt + 1
    plt.imshow(digits.images[i])
    plt.show()

#画像データのフォーマットを見る
print(digits.images[1].shape)
print(digits.images[1])
plt.imshow(digits.images[1])
plt.show()

#データの学習
X = digits.data
y = digits.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)

parameters = {'kernel':('linear','rbf','poly'),'C':[0.1,1.0,10.0],'gamma':[0.01,0.1,1.0,10.0],'decision_function_shape':('ovo','ovr')}
svc = svm.SVC()
clf = GridSearchCV(svc,parameters,scoring='accuracy',cv=5)
clf.fit(X_train,y_train)

#GridSearchで一番精度の良かった分類器を使う
clf_best = clf.best_estimator_

#テストデータを用いた予測を行う
y_test_pred = clf_best.predict(X_test)
print('accuracy = ',accuracy_score(y_test,y_test_pred))
print(classification_report(y_test,y_test_pred))

#適当なidを与えて、数字を予測させる
id = 11
dat = np.array([X_test[id]])
print('predicted number is ',clf_best.predict(dat))
print('Real number is ',y_test[id])
plt.matshow(X_test[id].reshape(8,8))
plt.show()