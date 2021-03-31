#偏微分のアルゴリズムを考える
# f(x,y) = x**2 + y**2 の(3,4)における偏微分を求める
def numerical_diff(f,x):
    h = 1e-4 #hをゼロに近づけるのだが、10^-4程度でも十分精度がでる
    return (f(x+h)-f(x-h)) / (2*h)

#x偏微分を求める
def function_tmp1(x0):
    return x0**2 + 4.0**2

print(numerical_diff(function_tmp1,3.0))

#y偏微分を求める
def function_tmp2(x1):
    return 3.0**2 + x1**2 

print(numerical_diff(function_tmp2,4.0))