import sys,os
sys.path.append(os.pardir) #親ディレクトリーのファイルをインポート
from mnist import load_mnist
from PIL import Image
import numpy as np 

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(X_train,t_train),(X_test,t_test) = load_mnist(flatten=True,normalize=False)

img = X_train[0]
label = t_train[0]
print(label)
print(X_test.shape)

print(img.shape) #(784,)
img = img.reshape(28,28) #形状を元の画像サイズに変換
print(img.shape)

img_show(img)