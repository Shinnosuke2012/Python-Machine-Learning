from sklearn.base import clone
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier

#SBS逐次次元削減法
class SBS():
    def __init__(self,estimator,k_features,scoring=accuracy_score,test_size=0.25,random_state=1):
        self.estimator = clone(estimator) #推定器をコピー
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)
        #分類をするためには、データを標準化しておくことが望ましい
        X_train_std = (X_train - X_train.mean()) /X_train.std()
        X_test_std = (X_test - X_test.mean()) /X_test.std()

        #全ての特徴量の個数、列インデックス
        dim = X_train.shape[1] #特徴量の個数
        self.indices_ = tuple(range(dim)) #0からdim-1までの数が定義されている
        self.subsets_ = [self.indices_] #0からdim-1までの配列となる
        #全ての特徴量を用いてスコアを計算する
        score = self._calc_score(X_train,y_train,X_test,y_test,self.indices_) #始めは全ての特徴量を使ってスコアを計算する
        self.scores_ = [score] #配列に置き換える

        #指定した特徴量の個数になるまで処理を継続する
        while dim>self.k_features:
            scores = []
            subsets = []

            #特徴量の部分集合を表すインデックスの組み合わせ事に処理を反復
            for p in combinations(self.indices_,r=dim-1): #次元の数を削減していく(dim-1個の要素をもつ配列をdim個用意する)
                score = self._calc_score(X_train,y_train,X_test,y_test,p)
                scores.append(score) #dim個のスコアが配列として格納される
                subsets.append(p)
            
            #最良のスコアのインデックスを抽出する
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            #特徴量の個数を一つだけ減らして次へ行く
            dim -= 1

            #スコアを格納する
            self.scores_.append(scores[best])
        
        #最後に格納したスコア
        self.k_score = self.scores_[-1]

        return self 
    
    def transform(self,X):
        #抽出した特徴量を返す
        return X[:,self.indices_]
    
    def _calc_score(self,X_train,y_train,X_test,y_test,indices):
        #指定された列番号indicesの特徴量を抽出してモデルに適合させる
        self.estimator.fit(X_train[:,indices],y_train)
        y_pred = self.estimator.predict(X_test[:,indices])
        score = self.scoring(y_test,y_pred)
        return score
    
#Wineデータセットの分析を行う
wine = datasets.load_wine()
feature_names = wine.feature_names

X = wine.data #特徴量
y = wine.target #正解ラベル

knn  = KNeighborsClassifier(n_neighbors=2)

sbs = SBS(knn,k_features=1)

sbs.fit(X,y)

#近傍点の個数のリスト(次元の数)
k_feat = [len(k) for k in sbs.subsets_]
#横軸を近傍点の個数、縦軸をスコアとした折れ線グラフをプロット
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.1])
plt.ylabel("Accuracy")
plt.xlabel("Number of features")
plt.title("Maximum score is "  + str(np.max(sbs.scores_)))
plt.grid(True)
plt.show()

#検証用データで一番スコアが高かった6の特徴量が何であったか確認する
k6 = list(sbs.subsets_[7])
for k in k6:
    print([feature_names[k]])