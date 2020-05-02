# -*- coding: UTF-8 -*-
"""
Python Basic Math Libraries
David L. Xu
Version 1.0.0
Adapt Python Version 3.7+
Started From July 7, 2019
Edited at Sept 17, 2019
There are alreay bunch of math libraries using C
But how can we implement it ourselves?
Written Purely in Python from Ground up
To kill time in summer holidays

NOTICE: This library is only for entertainment and 
do not ensure the precision to be tolerated

Simply use basic operations(+-*/) and some language
properties to achieve most of the functions;
Math is magic!

Tips:
1.In Python2, divide(/) two numbers get an integer
2.To avoid ambiquity, please run the script in python3
"""

###### basics mathematics ######
# absolute values
pi = 3.14159265358979
e = 2.718281828459
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

###some is—functions###

def is_even(n):
    if n%2==0:
        return True
    else:
        return False

def is_odd(n):
    if n%2==1:
        return True
    else:
        return False

def is_prime(n):
    pass

def is_decimal(n):
    if not is_odd(n) and not is_even(n):
        return True
    else:
        return False

def is_integer(n):
    if is_odd(n) or is_even(n):
        return True
    else:
        return False




###### some basic functions ######
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
###### trigonomitry functions ######
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
    
 
###### exponential and logarithm
def exp(x):
    
    return (1+x/100000.0)**100000

def ln(x):
    n = 1000000.0 # 数大了能算准，但是画图出现锯齿
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

###### linear algebra ######
#generates matrix with all zeros
def zeros(row,col):
    '''
    a=[]
    for i in range(row):
        a.append([])
        for j in range(col):
            a[i].append(0)
    return a
    '''
    return [[0 for i in range(col)]for i in range(row)]

#generates matrix with all ones
def ones(row,col):
    '''
    a=[]
    for i in range(row):
        a.append([])
        for j in range(col):
            a[i].append(1)
    return a
    '''
    return [[1 for i in range(col)]for i in range(row)]

#generates identity matrix
def eyes(n):
    a = [[0 for i in range(n)]for i in range(n)]
    for i in range(n):
        a[i][i]=1
    return a

def transpose(A):
    B = zeros(len(A[0]),len(A))
    for i in range(len(A[0])):
        for j in range(len(A)):
            B[i][j] = A[j][i]
    return B

#perform matrix muliplication
def multiply(A,B):
    #check if col of A is equal to row of B
    if(len(B)!=len(A[0])):
        raise ValueError("COLUMN of first matrix must be EQUAL to the ROW of second matrix!")
    result_row = len(A)
    result_column = len(B[0])
    result = [[0 for i in range(result_column)]for j in range(result_row)]
    
    common_rowcol = len(B)
    #use traditional row times columns;
    #but other ways can perform this like MIT 16.04 Gilbert Strang
    for i in range(result_row):
        for j in range(result_column):
            for k in range(common_rowcol):
                result[i][j] += A[i][k]*B[k][j]
    
    return result

#calculates the permutation of a set, used for calculating determinants
def permutation_sign(array):
    n = len(array)
    num = 0
    i,j = 0,1
    for i in range(n-1):
        for j in range(i+1,n):
            if array[i]>array[j]:
                num+=1;
    #return num
    if is_even(num):
        return 1
    elif is_odd(num):
        return -1
def permutation(array):
    n = len(array)
    num = 0
    i,j = 0,1
    for i in range(n-1):
        for j in range(i+1,n):
            if array[i]>array[j]:
                num+=1;
    return num
    
#返回一个数组的全排列，形式为二维数组，具体算法原理改日探讨
perm_num=0;
perm_list = []
# next time use perm, remember to reset the variables
def reset_perm(): # make a copy of the perm_list before reset
    perm_num=0;
    perm_list = []
def perm(array, k, m): # For the 1st time, use directly without reset, 
                    # but do use for the next time, otherwise newly generated one appended to the original one
    global perm_list
    global perm_num
    if(k > m):
        perm_list.append([])
        #print("permutation",perm_num+1)
        for i in range(m+1):
            perm_list[perm_num].append(array[i])
            #print("%d  "%array[i],end='');
        #print("\n\n");
        perm_num+=1
 
    else:
        for i in range(k,m+1):
            array[k],array[i]=array[i],array[k]
            
            perm(array, k + 1, m);
            array[k],array[i]=array[i],array[k]
    return perm_list       
        

#calculates the determinant of the matrix
def det(A):
    '''
    there are several ways to calculate the determinant
    1.use permutations denote the sign and product of deferent rows and columns
    2.transform into upper triangular matrix, get the product of the diagonal
    3.expand determinant recursively
    '''
    #the third method is preferred as computer programs. 
    
    #method 1
    n = len(A) # 行数
    m = len(A[0]) # 列数
    if n!=m:
        raise ValueError("input NOT SQUARE matrix!")
    row = [i for i in range(n)] # 生成从0到n-1的列表，用于固定行标
    col = perm(row,0,n-1)
    p = 1 # 用于每一小项累乘
    s = 0 # 行列式多项式求和
    for k in range(factorial(n)):
        #print("term",k+1)
        for (i,j) in transpose([row,col[k]]):
            #print("item_",i,j)
            p*=A[i][j]
            
        s+=(permutation_sign(col[k])*p)
        p=1
        #print("permutation_sign:",permutation_sign(col[k]))
    return s







# PLAYGROUND. run and play, which is the fun part!

import matplotlib.pyplot as plt
import numpy as np
import math




