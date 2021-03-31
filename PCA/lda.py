#線形判別分析による教師ありデータ圧縮
from sklearn import datasets 
from sklearn.model_selection import train_test_split
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression

class LDA():
    def __init__(self):
        pass 

    def fit(self,X,y):
        X_std = (X - X.mean()) /X.std()
        X_std = 100*X_std #あまりにも値が小さくなるためスケーリングを施す

        #平均ベクトルを計算する
        np.set_printoptions(precision=4)
        self.mean_vecs = []
        for label in range(1,4):
            self.mean_vecs.append(np.mean(X_std[y==label],axis=0))
            print('MV %s: %s\n' %(label,self.mean_vecs[label -1]))


        #Xの共分散行列 S_i = sigma(x in D_i)[(x - m_i)(X - m_i).T] /N_i を計算
        d = 13 
        self.S_W = np.zeros((d,d))
        for label in range(1,4):
            self.class_scatter = np.cov(X_std[y==label].T)
            self.S_W += self.class_scatter

        #クラス間変動行列S_Bを計算
        self.mean_oveall = np.mean(X_std,axis=0)
        d = 13
        self.S_B = np.zeros((d,d))
        for i,self.mean_vec in enumerate(self.mean_vecs):
            n = X_std[y==i+1,:].shape[0]
            self.mean_vec = self.mean_vec.reshape(d,1)
            self.mean_overall = self.mean_oveall.reshape(d,1)
            self.S_B += n*(self.mean_vec - self.mean_oveall).dot((self.mean_vec - self.mean_oveall).T)

        #固有値問題 S_W^-1 S_B w = λw を解く
        self.eigen_vals,self.eigen_vecs = np.linalg.eig(np.linalg.inv(self.S_W).dot(self.S_B))
        self.eigen_pairs = [(self.eigen_vals[i],self.eigen_vecs[:,i]) for i in range(len(self.eigen_vals))]
        self.eigen_pairs = sorted(self.eigen_pairs,key=lambda k: k[0], reverse=True)
        print('Eigenvalues in decreasing order:\n')
        for eigen_val in self.eigen_pairs:
            print(eigen_val[0])

        #固有値からそれぞれの寄与率を求める
        self.tot = sum(self.eigen_vals.real)
        #分散説明率とその累積和
        self.discr = [(i/self.tot) for i in sorted(self.eigen_vals.real,reverse=True)]
        self.cum_discr = np.cumsum(self.discr)

        #分散説明率のグラフ作成
        plt.bar(range(1,14),self.discr,alpha=0.5,align="center",label="individual discriminability")

        #累積分散説明率のグラフを作成
        plt.step(range(1,14),self.cum_discr,where="mid",label="cumulative discriminability")

        plt.xlabel("Linear Discriminants")
        plt.ylabel("Discriminability ratio")
        plt.legend(loc="best")
        plt.show()

        #最も特徴説明率のあるベクトルを用いて変換行列を作成
        self.w_ = np.hstack((self.eigen_pairs[0][1][:,np.newaxis].real,self.eigen_pairs[1][1][:,np.newaxis].real))
        print('Matrix w:\n',self.w_)

        self.X_train_lda = X_std.dot(self.w_)

    def fit_transform(self,X,y):
        return self.X_train_lda
    
    def plot(self,X,y):
        colors = ['r','b','g']
        markers = ['s','x','o']
        for l,c,m in zip(np.unique(y),colors,markers):
            plt.scatter(self.X_train_lda[y_train==l,0],self.X_train_lda[y_train==l,1],c=c,marker=m,label=l)

        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.legend(loc='lower right')
        plt.show()

#Wineデータセットの分析を行う
wine = datasets.load_wine()
feature_names = wine.feature_names

X = wine.data #特徴量
y = wine.target #正解ラベル
X = X
y = y + 1

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

#インスタンスの生成を行う
lda = LDA()
lda.fit(X_train,y_train)
lda.plot(X_train,y_train)

#線形分離を行う
X_train_lda = lda.fit_transform(X_train,y_train)
from mlxtend import plot_decision_regions
Lr = LogisticRegression(penalty='l2')
Lr.fit(X_train_lda,y_train)
plot_decision_regions(X_train_lda,y_train,classifier=Lr)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc="lower right")
plt.show()