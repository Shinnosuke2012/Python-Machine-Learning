#数値微分のアルゴリズムを考える
def function(x):
    return x**2 + x
def numerical_diff(f,x):
    h = 1e-4 #hをゼロに近づけるのだが、10^-4程度でも十分精度がでる
    return (f(x+h)-f(x-h)) / (2*h)

print(numerical_diff(function,5))

import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(0.0,10.0,0.1)
y = function(x)
diff = numerical_diff(function,x)
plt.figure(figsize=(8,6))
plt.plot(x,y,c='blue',label='original function')
plt.plot(x,diff,c='red',label='defferenciated function')
plt.legend(loc='lower right')
plt.show()