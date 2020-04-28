# -*- coding: UTF-8 -*-
"""
Python Basic Math Libraries
David L. Xu
Version 1.0.0
Adapt Python Version 3.7+
Started From July 7, 2019

There are alreay bunch of math libraries using C
But how can we implement it ourselves?
Written Purely in Python from Ground up
To Find things to do In Summer holidays

NOTICE: This library is only for entertainment and 
do not ensure the precision to be tolerated

Simply use basic operations(+-*/) and some language
properties to achieve most of the functions;
Math is magic!

"""
### basics
# absolute values
pi = 3.14159265358979
def abs(x):
    if x>=0:
        return x 
    else:
        return -x

def floor(x):
    if x>=0:
        return int(x)
    else:
        return int(x)-1
  
def ceil(x):
    if x>=0:
        return int(x)+1
    else:
        return int(x)

def round(x):
    if x>=0:
        if x-int(x)>=0.5:
            return int(x)+1
        else:
            return int(x)
    else:
        if int(x)-x>=0.5:
            return int(x)-1
        else:
            return int(x)

def factorial(x):
    if x == 0 or x == 1:
        return 1
    else:
        return x*factorial(x-1)
# 隔项阶乘
def factorial_2(x):

    if(x==1 or x == 0):
        return 1
    else:
        return x*factorial_2(x-2);

#开根号，二分法
def sqrt_binarysearch(x):
    min_val = 0
    max_val = x
    i=0

    while True:
        ans = (max_val + min_val)/2.0
        #print(ans,max_val,min_val)
        if abs(ans**2-x)<1e-12:
            return ans
        if ans**2 > x:
            max_val = ans
        elif ans**2 < x:
            min_val = ans
        


#开根号，牛顿迭代法，更快
def sqrt_newton(x):
    if(x <= 0):
        return 0;
    res = x   
    lastres = 0
    while(abs(lastres-res)>1e-8):
        lastres = res;
        res = (res + x/res)/2.0;
    
    
 
    return res;

#暂且把牛顿迭代法作为官方迭代法
def sqrt(x):
    return sqrt_newton(x)
'''
用泰勒展开求根号没有希望，因为后来误差会越来越大
def sqrt(x):
    temp = 1
    fac_odd = 1
    fac_even = 1
    for n in range(1,30):
        #fac_odd = fac_odd*(2*n-1)
        #print("odd",fac_odd)
        #print("even",fac_even)
        #fac_even = fac_even*2*n
        temp = temp + (-1)**(n+1)*(factorial_2(2*n-1)/factorial_2(2*n))*x**n
    return temp
'''
'''
def sqrt(x):

    temp=1
    temp_n = 1
    for i in range(1,20):
        if i == 1:
            temp_n = 1
        else:
            temp_n=temp_n*(2*i-3)
        temp = temp + (-1)**(i+1)*(temp_n)/(2**i)/factorial(i)*(x-1)**i
    return temp
'''
### trigonomitry functions
# sin(x) using Taylor expansions
def sin(x):
    x = x % (2*pi) # transform any value into [0,2pi] to avoid loss in precision in finite Taylor expansions
    temp = 0
    for i in range(1,20):
        temp = temp + ((-1)**(i-1))*(x**(2*i-1))/factorial(2*i-1)
    return temp

def cos(x):
    x = x % (2*pi)
    temp = 1
    for i in range(1,20):
        temp = temp + ((-1)**(i))*(x**(2*i))/factorial(2*i)
    return temp#float("%.3f"%temp)
    
def tan(x):
    return sin(x)/cos(x)
    
def arcsin(x):
    pass
    
def arccos(x):
    pass
    
def arctan(x):
    pass
    
 

def exp(x):
    
    return (1+x/100000.0)**100000

def ln(x):
    n = 10000000000000000.0 # 数大了能算准，但是画图出现锯齿
    return n * ((x ** (1/n)) - 1)

def log10(x):
    return ln(x)/ln(10)


def derivative(f,a,method='central',h=0.0001):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h            
    '''
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)#float("%.3f" % ((f(a + h) - f(a - h))/(2*h)))
    elif method == 'forward':
        return (f(a + h) - f(a))/h#float("%.3f" % ((f(a + h) - f(a))/h))
    elif method == 'backward':
        return (f(a) - f(a - h))/h#float("%.3f" % ((f(a) - f(a - h))/h))
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")





# PLAYGROUND. run and play, which is the fun part!

import matplotlib.pyplot as plt
import numpy as np
import math


#fig, ax = plt.subplots()
#ax.set_ylim(-3, 3)

'''
x = np.linspace(0.5,99,10000)

plt.plot(x,ln(x))
plt.plot(x,ln(x)-np.log(x))
plt.show()
'''
print(ln(1000000))

