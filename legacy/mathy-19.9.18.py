# -*- coding: UTF-8 -*-
"""
Python Basic Math Libraries
David L. Xu
Version 1.0.0
Adapt Python Version 3.7+
Started From July 7, 2019
Edited at Sept 18, 2019
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
2.To avoid ambiquity, please run the script in Python3
"""

###### basics mathematics ######
# absolute values
pi = 3.1415926535897932384626
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


def combination(n,m): # order: selet_item,total_item
    if n>m:
        raise ValueError("the first num must be LESS than of EQUAL to the second num")
    return factorial(m)/factorial(n)/factorial(m-n)

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

#这种转置方式有问题
def transpose(A):
    B = zeros(len(A[0]),len(A))
    #print("after_shape:",len(A[0]),len(A))
    for i in range(len(A[0])):
        #print("i:",i)
        for j in range(len(A)):
            #print("j:",j)
            B[i][j] = A[j][i]
    return B

def transpose_square(A):
    B = zeros(len(A),len(A))
    for i in range(len(A)):
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
#计算排列的逆序数，用于线性代数元素的符号判定
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
perm_count=0
# next time use perm, remember to reset the variables
def reset_perm(): # make a copy of the perm_list before reset
    global perm_num
    perm_num=0
    global perm_list
    perm_list = []
    global perm_count
    perm_count = 0

#用于生成给定元素的全排列
#每次递归调用一套perm生成全排列之前，手动调用一下reset_perm()，不然上次全排列的结果会遗留
"""
自己写的函数，代替方案是现成库函数
from itertools import combinations, permutations
print(list(permutations([1,2,3],3)))
"""
def perm(array, k, m): # For the 1st time, use directly without reset, 
                    # but do use for the next time, otherwise newly generated one appended to the original one
    global perm_list
    global perm_num
    global perm_count
    perm_count+=1
    
    #if(perm_count > factorial(len(array))):
        #pass#reset_perm()#想办法自动重置一下，但是找不到自动重置的时机，没有切入点
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
    reset_perm()# 每次递归生成perm之前，手动清零一次
    col = perm(row,0,n-1)
    p = 1 # 用于每一小项累乘
    s = 0 # 行列式多项式求和
    for k in range(factorial(n)):
        #print("term",k+1)
        for (i,j) in transpose([row,col[k]]):   #这种来自不同行不同列的算法有点问题
            #print("item_",i,j)
            p*=A[i][j]
        s+=(permutation_sign(col[k])*p)
        p=1
        #print("permutation_sign:",permutation_sign(col[k]))

    return s



def deepcopy(A):
    B = zeros(len(A),len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            B[i][j] = A[i][j]
    """
    C = A
    print(A,id(A),id(A[0]),id(A[0][0]))
    print(B,id(B),id(B[0]),id(B[0][0]))
    print(C,id(C),id(C[0]),id(C[0][0]))
    """
    return B
#import copy  #自己写一个deepcopy函数，这样就不用调用这唯一的库函数了
def minor(A,row,col):
    B = deepcopy(A) #如果直接复制，会把传进来的A给修改了 浅复制二维数组的每个元素指向的行向量是一个地址
    for i in range(len(B)):
        del B[i][col]
    del B[row]
    return (-1)**(row+col)*det(B)

# calculates the adjoint of a given matrix
def adjoint(A):    
    B = deepcopy(A)
    C = zeros(len(A),len(A[0]))
    row = len(A)
    col = len(A[0])
    for i in range(row):
        for j in range(col):
            C[i][j]=minor(B,j,i)

    return C
# calculates the inverse of a given matrix
def inv(A):
    B = adjoint(A)
    for i in range(len(B)):
        for j in range(len(B)):
            B[i][j]=B[i][j]/det(A)
    return B

def rank(A):
    # convert to row echelon is complex, thus use direct calculation of the 


# 用此函数形成增广矩阵 未完成
def comb_col(*matrices): #接受之后，matrices是一个三维列表   
    M = []
    for i in range(len(matrices[0])):
        M.append([])
    for k in range(len(matrices[0])):
        for i in range(len(matrices)):
            M[k].extend(matrices[i][k])
    return M
def comb_row(*matrices):
    M = []
    for k in range(len(matrices)):
        for i in range(len(matrices[k])):
            M.append(matrices[k][i])
    return M

# as for return value, the first index refers to num of rows; the second num of cols
def matrix_shape(A):
    return (len(A),len(A[0]))
# 等时机合适，引入矩阵对象，自带shape，维度，行列式，逆等特征

# raise error when infinite solutions
def solve_linear_equation(A,b):
    return multiply(inv(A),b)

def solve_augmented_mat(A):
    B = deepcopy(A)
    b = zeros(len(A),1)
    for i in range(len(A)):
        b[i][0] = A[i][len(A[0])-1]
        B[i].pop()
    #print(B)
    #print(b)
    return solve_linear_equation(B,b)  
   
# PLAYGROUND. run and play, which is the fun part!
# Tips: row vectors should be written as [[1,2]]; column vectors written as [[1],[2]]
#import matplotlib.pyplot as plt
#import numpy as np
#import math

# 修改意见：传进去的矩阵返回的时候不要把原矩阵改了，看看指针引用的问题

'''
M=[[[1,2],
    [3,4]],

    [[5,6],
    [7,8]],

    [[9,10],
    [11,12]]]

A = [[1,2],[3,4]]
B = [[5,6],[7,8]]
C = comb_col(A,B)
D = comb_row(A,B)
print(C,matrix_shape(C))
print(D,matrix_shape(D))
'''
A = [ 
        [1,-1,-1,1,1,1],
        [2,-1,-3,2,2,2],
        [3,2,-5,3,3,3],
        [7,5,4,3,5,1],
        [2,2,2,5,75,4],
        [2,7,6,4,4,3]
    ]
rank(A)
print(det(A))
'''
print((A))
B = deepcopy(A)
print(A,id(A),id(A[0]),id(A[0][0]))
print(B,id(B),id(B[0]),id(B[0][0]))
b = transpose([[2,1,0]])
#print(solve_linear_equation(A,b))

C = [[1,-1,-1,2],[2,-1,-3,1],[3,2,-5,0]]
print(C)
print(solve_augmented_mat(C))
print(C)
'''
#二项式定理的系数分布散点图，可以用中英文写一些数学小现象，用好latex

