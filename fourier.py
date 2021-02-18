from imports import *
#from sympy import *
import numpy as np
import matplotlib.pyplot as plt


# fourier transform
def dft_cos(x,n):
    '''
    借鉴了思路 https://www.zhihu.com/question/21314374/answer/542909849
    '''
    X = [0 for i in range(n)]
    for k in range(n):
        for i in range(n):
            X[k]+=x[i]*cos(2*pi*k*i/n)
    return X


#w = dft_cos(y,40)
#print(w)

def dft(x,n):
    """
    利用复数存储dft结果
    """
    X = [0 for i in range(n)]
    for k in range(n):
        for i in range(n):
            X[k]+=x[i]*e**(-2*pi*1j*k*i/n)
    return X

def dft_cos_sin(x,n):
    """
    利用cos和-sin来存储结果，注意sin为正，不是dft中-sin了
    """
    X_cos = [0 for i in range(n)]
    X_sin = [0 for i in range(n)]
    for k in range(n):
        for i in range(n):
            #X[k]+=x[i]*e**(-2*pi*1j*k*i/n)
            X_cos[k]+=x[i]*cos(2*pi*k*i/n)
            X_sin[k]+=x[i]*sin(2*pi*k*i/n)
    return X_cos,X_sin  




# 以下来自https://www.jianshu.com/p/0bd1ddae41c4
"""
@Author: Sam
@Function: Fast Fourier Transform
@Time: 2020.02.22 16:00
"""
from cmath import sin, cos, pi

class FFT_pack():
    def __init__(self, _list=[], N=0):  # _list 是传入的待计算的离散序列，N是序列采样点数，对于本方法，点数必须是2^n才可以得到正确结果
        self.list = _list  # 初始化数据
        self.N = N
        self.total_m = 0  # 序列的总层数
        self._reverse_list = []  # 位倒序列表
        self.output =  []  # 计算结果存储列表
        self._W = []  # 系数因子列表
        for _ in range(len(self.list)):
            self._reverse_list.append(self.list[self._reverse_pos(_)])
        self.output = self._reverse_list.copy()
        for _ in range(self.N):
            self._W.append((cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** _)  # 提前计算W值，降低算法复杂度

    def _reverse_pos(self, num) -> int:  # 得到位倒序后的索引
        out = 0
        bits = 0
        _i = self.N
        data = num
        while (_i != 0):
            _i = _i // 2
            bits += 1
        for i in range(bits - 1):
            out = out << 1
            out |= (data >> i) & 1
        self.total_m = bits - 1
        return out

    def FFT(self, _list, N, abs=True) -> list:  # 计算给定序列的傅里叶变换结果，返回一个列表，结果是没有经过归一化处理的
        """参数abs=True表示输出结果是否取得绝对值"""
        self.__init__(_list, N)
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)
            num_each = self.N // _split
            for _ in range(_split):
                for __ in range(num_each // 2):
                    temp = self.output[_ * num_each + __]
                    temp2 = self.output[_ * num_each + __ + num_each // 2] * self._W[__ * 2 ** (self.total_m - m - 1)]
                    self.output[_ * num_each + __] = (temp + temp2)
                    self.output[_ * num_each + __ + num_each // 2] = (temp - temp2)
        if abs == True:
            for _ in range(len(self.output)):
                self.output[_] = self.output[_].__abs__()
        return self.output

    def FFT_normalized(self, _list, N) -> list:  # 计算给定序列的傅里叶变换结果，返回一个列表，结果经过归一化处理
        self.FFT(_list, N)
        max = 0   # 存储元素最大值
        for _ in range(len(self.output)):
            if max < self.output[_]:
                max = self.output[_]
        for _ in range(len(self.output)):
            self.output[_] /= max
        return self.output

    def IFFT(self, _list, N) -> list:  # 计算给定序列的傅里叶逆变换结果，返回一个列表
        self.__init__(_list, N)
        for _ in range(self.N):
            self._W[_] = (cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** (-_)
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)
            num_each = self.N // _split
            for _ in range(_split):
                for __ in range(num_each // 2):
                    temp = self.output[_ * num_each + __]
                    temp2 = self.output[_ * num_each + __ + num_each // 2] * self._W[__ * 2 ** (self.total_m - m - 1)]
                    self.output[_ * num_each + __] = (temp + temp2)
                    self.output[_ * num_each + __ + num_each // 2] = (temp - temp2)
        for _ in range(self.N):  # 根据IFFT计算公式对所有计算列表中的元素进行*1/N的操作
            self.output[_] /= self.N
            self.output[_] = self.output[_].__abs__()
        return self.output

    def DFT(self, _list, N) -> list:  # 计算给定序列的离散傅里叶变换结果，算法复杂度较大，返回一个列表，结果没有经过归一化处理
        self.__init__(_list, N)
        origin = self.list.copy()
        for i in range(self.N):
            temp = 0
            for j in range(self.N):
                temp += origin[j] * (((cos(2 * pi / self.N) - sin(2 * pi / self.N) * 1j)) ** (i * j))
            self.output[i] = temp.__abs__()
        return self.output





if __name__ == '__main__':
    #list = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]
    #a = FFT_pack().FFT(list, 16, False)
    #print(a)
    num = 40
    x = [i for i in range(num)]
    #y = list(map(lambda n:cos(2*pi*(2*n)/40+pi/10)+0.5*sin(5*n),x))
    y = list(map(lambda n:cos(2*pi*(2*n)/num),x))


    w_cos,w_sin = dft_cos_sin(y,num)
    plt.scatter(x,w_cos,color='orange')
    plt.scatter(x,w_sin,color='pink')
    plt.plot(x,y)
    plt.show()