# -*- coding: UTF-8 -*-
"""
Python Basic Math Libraries
David L. Xu
Version 1.0.0
Adapt Python Version 3.7+
Started From July 7, 2019
First uploaded on my Github acount on Oct 15, 2019
There are alreay bunch of math libraries using C
But how can we implement it ourselves?
Written Purely in Python from Ground up
For recreational use in summer holidays

NOTICE: This library is only for entertainment and 
does not ensure the precision to be tolerated

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
'''
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
'''
def min(*list):
    t = list[0]
    for i in range(len(list)):
        if t>list[i]:
            t = list[i]
    return t
        
def max(*list):
    t = list[0]
    for i in range(len(list)):
        if t<list[i]:
            t = list[i]
    return t
    
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

def is_zero(n):
	if n == 0:
		return True
	else:
		return False
def is_one(n):
	if n == 1:
		return True
	else:
		return False
def is_nonzero(n):
	if n != 0:
		return True
	else:
		return False
def is_positive(n):
	if n > 0:
		return True
	else:
		return False
def is_negative(n):
	if n < 0:
		return True
	else:
		return False
def is_nonnegative(n):
	if n >= 0:
		return True
	else:
		return False
def is_nonpositive(n):
	if n <= 0:
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

def log(x,y): #logarithm of y on the base of x

    return ln(y)/ln(x)

def pow(x,y):
    # y 可以是实数
    # 利用x^y=e^(ln(x)*y)
    # not precise because the exp() and ln() implementation is coarse
    if isinstance(y,float):
        return exp(ln(x)*y)
    else:     
        prod = 1
        if y >=0:
            for i in range(y):
                prod *= x
            return prod
        else:
            for i in range(-y):
                prod /= x
            return prod

print(pow(2,-10.5))
def derivative(f,a,method='central',h=0.00001):

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

# steps being the number of members between the interval
# conventionally, start value is smaller than end value

def linspace(start_val,end_val,steps = 50):
	
	#if 'numpy' in dir() or 'linspace' in dir(): 
	print("You are calling a built-in linspace instead of the numpy's. ",end = '')
	print("However, this may not be precise as you think")
	list = []
	i = start_val
	interval = (end_val-start_val)/steps
	while i < end_val:
		list.append(i)
		i+=interval
	return list

# warning! extremely not precise! only for recreational use
# substitutes with more powerful Simpson formula and Gauss formula
def integrate_regular(function,start,end,precision = 2500):
	sum = 0
	
	for i in linspace(start,end,precision):
		sum+=function(i)*(end-start)/precision
	return float("%.16f"%sum)
# simpson formula and trapezoid method belongs to newton cotes
def integrate_simpson():
	pass

def integrate_gauss(function,start,end,n=3):
    # 高斯点及其求积系数列表
    x1=[0.0]
    A1=[2]
    x2=[-0.5773502691896250, 0.5773502691896250]
    A2=[1, 1]
    x3=[-0.7745966692414830, 0.0, 0.7745966692414830]
    A3=[0.5555555555555550, 0.8888888888888880, 0.5555555555555550]
    x4=[0.3399810435848560, -0.3399810435848560, 0.8611363115940520, -0.8611363115940520]
    A4=[0.6521451548625460, 0.6521451548625460, 0.3478548451374530, 0.3478548451374530]
    x5=[0.0, 0.5384693101056830, -0.5384693101056830, 0.9061798459386640, -0.9061798459386640]
    A5=[0.5688889, 0.4786287, 0.4786287, 0.2369269, 0.2369269]
    x6=[-0.9324695142031520,-0.6612093864662640,-0.2386191860831960,0.2386191860831960,0.6612093864662640,0.9324695142031520]
    A6=[0.1713244923791700, 0.3607615730481380, 0.4679139345726910, 0.4679139345726910,0.3607615730481380, 0.1713244923791700]
    x7=[-0.949107912,-0.741531186,-0.405845151,0,0.405845151,0.741531186,0.949107912]
    x8=[-0.960289856,-0.796666477,-0.52553241,-0.183434642,0.183434642,0.52553241,0.796666477,0.960289856]
    x9=[-0.96816024,-0.836031107,-0.613371433,-0.324253423,0,0.324253423,0.613371433,0.836031107,0.96816024]
    x10=[-0.973906529,-0.865063367,-0.679409568,-0.433395394,-0.148874339,0.148874339,0.433395394,0.679409568,0.865063367,0.973906529]
    x11=[-0.978228658,-0.8870626,-0.730152006,-0.519096129,-0.269543156,0,0.269543156,0.519096129,0.730152006,0.8870626,0.978228658]
    x12=[-0.981560634,-0.904117256,-0.769902674,-0.587317954,-0.367831499,-0.125233409,0.125233409,0.367831499,0.587317954,0.769902674,0.904117256,0.981560634]
    x13=[-0.984183055,-0.917598399,-0.801578091,-0.642349339,-0.448492751,-0.230458316,0,0.230458316,0.448492751,0.642349339,0.801578091,0.917598399,0.984183055]
    x14=[-0.986283809,-0.928434884,-0.827201315,-0.687292905,-0.515248636,-0.319112369,-0.108054949,0.108054949,0.319112369,0.515248636,0.687292905,0.827201315,0.928434884,0.986283809,]
    x15=[-0.987992518,-0.937273392,-0.848206583,-0.724417731,-0.570972173,-0.394151347,-0.201194094,0,0.201194094,0.394151347,0.570972173,0.724417731,0.848206583,0.937273392,0.987992518]
    x16=[-0.989400935,-0.944575023,-0.865631202,-0.755404408,-0.617876244,-0.458016778,-0.281603551,-0.09501251,0.09501251,0.281603551,0.458016778,0.617876244,0.755404408,0.865631202,0.944575023,0.989400935]
    x17=[-0.990575475,-0.950675522,-0.880239154,-0.781514004,-0.657671159,-0.512690537,-0.351231763,-0.178484181,0,0.178484181,0.351231763,0.512690537,0.657671159,0.781514004,0.880239154,0.950675522,0.990575475]
    x18=[-0.991565168,-0.95582395,-0.892602466,-0.803704959,-0.691687043,-0.559770831,-0.411751161,-0.251886226,-0.084775013,0.084775013,0.251886226,0.411751161,0.559770831,0.691687043,0.803704959,0.892602466,0.95582395,0.991565168]
    x19=[-0.992406844,-0.960208152,-0.903155904,-0.822714657,-0.720966177,-0.600545305,-0.464570741,-0.3165641,-0.160358646,0,0.160358646,0.3165641,0.464570741,0.600545305,0.720966177,0.822714657,0.903155904,0.960208152,0.992406844]
    x20=[-0.993128599,-0.963971927,-0.912234428,-0.839116972,-0.746331906,-0.636053681,-0.510867002,-0.373706089,-0.227785851,-0.076526521,0.076526521,0.227785851,0.373706089,0.510867002,0.636053681,0.746331906,0.839116972,0.912234428,0.963971927,0.993128599]
    A7=[0.129484966,0.279705391,0.381830051,0.417959184,0.381830051,0.279705391,0.129484966]
    A8=[0.101228536,0.222381034,0.313706646,0.362683783,0.362683783,0.313706646,0.222381034,0.101228536]
    A9=[0.081274388,0.180648161,0.260610696,0.312347077,0.330239355,0.312347077,0.260610696,0.180648161,0.081274388]
    A10=[0.066671344,0.149451349,0.219086363,0.269266719,0.295524225,0.295524225,0.269266719,0.219086363,0.149451349,0.066671344]
    A11=[0.055668567,0.125580369,0.186290211,0.233193765,0.262804545,0.272925087,0.262804545,0.233193765,0.186290211,0.125580369,0.055668567]
    A12=[0.047175336,0.106939326,0.160078329,0.203167427,0.233492537,0.249147046,0.249147046,0.233492537,0.203167427,0.160078329,0.106939326,0.047175336]
    A13=[0.040484005,0.0921215,0.13887351,0.178145981,0.207816048,0.22628318,0.232551553,0.22628318,0.207816048,0.178145981,0.13887351,0.0921215,0.040484005]
    A14=[0.03511946,0.080158087,0.121518571,0.157203167,0.185538397,0.205198464,0.215263853,0.215263853,0.205198464,0.185538397,0.157203167,0.121518571,0.080158087,0.03511946]
    A15=[0.030753242,0.070366047,0.10715922,0.139570678,0.166269206,0.186161,0.198431485,0.202578242,0.198431485,0.186161,0.166269206,0.139570678,0.10715922,0.070366047,0.030753242]
    A16=[0.027152459,0.062253524,0.095158512,0.124628971,0.149595989,0.169156519,0.182603415,0.18945061,0.18945061,0.182603415,0.169156519,0.149595989,0.124628971,0.095158512,0.062253524,0.027152459]
    A17=[0.024148303,0.055459529,0.085036148,0.111883847,0.135136368,0.154045761,0.168004102,0.176562705,0.17944647,0.176562705,0.168004102,0.154045761,0.135136368,0.111883847,0.085036148,0.055459529,0.024148303]
    A18=[0.021616014,0.049714549,0.07642573,0.100942044,0.122555207,0.140642915,0.154684675,0.164276484,0.169142383,0.169142383,0.164276484,0.154684675,0.140642915,0.122555207,0.100942044,0.07642573,0.049714549,0.021616014]
    A19=[0.019461788,0.044814227,0.069044543,0.091490022,0.111566646,0.128753963,0.142606702,0.152766042,0.158968843,0.16105445,0.158968843,0.152766042,0.142606702,0.128753963,0.111566646,0.091490022,0.069044543,0.044814227,0.019461788]
    A20=[0.017614007,0.04060143,0.062672048,0.083276742,0.10193012,0.118194532,0.131688638,0.142096109,0.149172986,0.152753387,0.152753387,0.149172986,0.142096109,0.131688638,0.118194532,0.10193012,0.083276742,0.062672048,0.04060143,0.017614007]
    summation = 0
    
    if n == 1:
        p=x1
        t=A1
    elif n == 2:
        p = x2
        t = A2
    elif n == 3:
        p = x3
        t = A3
    elif n == 4:
        p = x4
        t = A4
    elif n == 5:
        p = x5
        t = A5
    elif n ==6:
        p = x6
        t = A6
    elif n ==7:
        p = x7
        t = A7
    elif n ==8:
        p = x8
        t = A8
    elif n ==9:
        p = x9
        t = A9
    elif n ==10:
        p = x10
        t = A10
    elif n ==11:
        p = x11
        t = A11
    elif n ==12:
        p = x12
        t = A12
    elif n ==13:
        p = x13
        t = A13
    elif n ==14:
        p = x14
        t = A14
    elif n ==15:
        p = x15
        t = A15
    elif n ==16:
        p = x16
        t = A16
    elif n ==17:
        p = x17
        t = A17
    elif n ==18:
        p = x18
        t = A18
    elif n ==19:
        p = x19
        t = A19
    elif n ==20:
        p = x20
        t = A20
    for i in range(n):
        summation +=function((end-start)*p[i]/2+(start+end)/2)*t[i]
    summation*=(end-start)/2
    return summation
def integrate():
    pass





'''
from sympy import *
x = symbols('x')
print(float(integrate(x*sin(x),(x,0,1))))
'''

# 暂时用列表的形式表示矩阵，考虑一下namedlist(类似namedtuple)

###### linear algebra ######
#print matrix in a more readable way
def print_matrix(A,precision=2,name = 'Matrix'):
    print(name+"[")
    for i in range(len(A)):
        print("\t",end='')
        for j in range(len(A[0])):  
                
            print(format(A[i][j],"."+str(precision)+"f"),end='\t')
        print()
    print(']') 
# auto judge row vector and column vector and print
def print_vector(A,precision=4):

    if(len(A) == 1):
        print("Row Vector[")
        print("\t",end='')
        for j in range(len(A[0])):
            print(format(A[0][j],"."+str(precision)+"f"),end='\t')
        print("\n]")
    elif(len(A[0]) == 1):
        print("Column Vector[")
        for i in range(len(A)):
            print("\t",format(A[i][0],"."+str(precision)+"f"))
        print("]")
    else:
        raise ValueError("NOT a vector!")
# print vector group
def print_vectors(*vec_tuple,precision=4):
    for k in range(len(vec_tuple)):
        if(len(vec_tuple[k]) == 1):
            print("Row Vector(%d)["%(k+1))
            print("\t",end='')
            for j in range(len(vec_tuple[k][0])):
                print(format(vec_tuple[k][0][j],"."+str(precision)+"f"),end='\t')
            print("\n]")
        elif(len(vec_tuple[k][0]) == 1):
            print("Column Vector(%d)["%(k+1))
            for i in range(len(vec_tuple[0])):
                print("\t",format(vec_tuple[k][i][0],"."+str(precision)+"f"))
            print("]")
        else:
            raise ValueError("NOT a vector!")
def print_vector_group(vec_tuple,precision=4):
    for k in range(len(vec_tuple)):
        if(len(vec_tuple[k]) == 1):
            print("Row Vector(%d)["%(k+1))
            print("\t",end='')
            for j in range(len(vec_tuple[k][0])):
                print(format(vec_tuple[k][0][j],"."+str(precision)+"f"),end='\t')
            print("\n]")
        elif(len(vec_tuple[k][0]) == 1):
            print("Column Vector(%d)["%(k+1))
            for i in range(len(vec_tuple[0])):
                print("\t",format(vec_tuple[k][i][0],"."+str(precision)+"f"))
            print("]")
        else:
            raise ValueError("NOT a vector!")
def nulls(col):
    a = []
    for i in range(col):
        a.append([])
    return a
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
def diag(*a):
    l = len(a)
    A = zeros(l,l)
    for i in range(l):
        for j in range(l):
            if i ==j:
                A[i][j] = a[i]
    return A
def randmat(row,col,largest = 10):#,property = "None"):
    import random
    A = zeros(row,col)
    for i in range(row):
        for j in range(col):
            A[i][j] = random.randint(0,largest)
    return A
def check_matrix_square(A):
    if len(A)==len(A[0]):
        return True
    else:
        return False
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

# matrix power, regular power see pow()
def power(A,k):

    if k != 0:
        k -= 1
        return multiply(power(A,k),A)
    if k == 0:
        return eyes(len(A))

#calculates the permutation of a set, used for calculating determinants
def permutation_sign(array):
    n = len(array)
    num = 0
    i,j = 0,1
    for i in range(n-1):
        for j in range(i+1,n):
            if array[i]>array[j]:
                num+=1
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
perm_num=0
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


# diy a function like copy.deepcopy() 
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



#### functions below changes the original input matrix, copy if necessary ###
# exchange two certain rows of a given matrix
def exchange_rows(B,r1,r2):
    #B=deepcopy(A)
    B[r1],B[r2]=B[r2],B[r1]
    return B

# exchange two certain columns of a given matrix
def exchange_cols(B,c1,c2):
    #B=deepcopy(A)
    for i in range(len(B)):
        B[i][c1],B[i][c2]=B[i][c2],B[i][c1]
    return B

# multiply a certain row by a given factor
def multiply_row(B,r,k): # r the row number, k the factor
    for i in range(len(B[0])):
        B[r][i] = k*B[r][i]
    return B

def add_rows(B,r1,r2): # adds r2 to r1
    for i in range(len(B[0])):
        B[r1][i] += B[r2][i]
    return B


def add_rows_by_factor(B,r1,k,r2): # r1+k*r2, adds k*r2 to r1
    #r1-=1
    #r2-=1
    for i in range(len(B[0])):
        B[r1][i] += k*B[r2][i]
    return B

def add_rows_by_factor_approximate(B,r1,k,r2,appr = 10): # r1+k*r2 adds k*r2 to r1
    #r1-=1
    #r2-=1
    for i in range(len(B[0])):
        B[r1][i] = round(B[r1][i],appr)
        b = round(k*B[r2][i],appr)
        B[r1][i]+=b
    return B
################# end section ################    
# to form augmented matrix using this
# 改进：维数不同时可以自动填零
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
def split_col(A):
    M = []
    for j in range(len(A[0])):
        N = []
        for i in range(len(A)):
            N.append([A[i][j]])
        M.append(N)
    return tuple(M)
def split_row(A):
    M = []
    for i in range(len(A)):
        M.append([A[i]])   # 把原来各自矩阵的一行转换成完整的二维矩阵
    return tuple(M) # 返回tuple,这样看起来像返回多个值，而不是三层list
#check i elements under a_ij is all zeros (a_ij included)
def below_all_zero(A,i,j):
    flag = 0
    for k in range(len(A)-i):
        if A[len(A)-k-1][j] == 0:
            flag+=0   
        
        else:
            flag+=1
    if flag>0:
        return False
    else:
        return True  
def 打印子式(A):
    from itertools import combinations
    rows = []
    for i in range(len(A)):
        rows.append(i)
    cols = []
    for i in range(len(A[0])):
        cols.append(i)
    n = min(len(A),len(A[0]))  
    while n>0:        
        new_rows =  list(combinations(rows,n))#包含了行所有的可能
        new_cols = list(combinations(cols,n))#包含了列所有的可能
        B = zeros(n,n)
        #print(new_rows)
        #print(new_cols)
        for a in range(len(new_rows)):
            for b in range(len(new_cols)):       
                for i in range(n):
                    for j in range(n):
                        B[i][j] = A[new_rows[a][i]][new_cols[b][j]]   
                print(B)
                #if det(B) != 0:
                    #return len(B)             
        n-=1
        #if n==0:
            #return 0
       
def rank(A):
    # convert to row echelon is complex, thus use direct calculation of the 最高阶非零子式
    from itertools import combinations
    rows = []
    for i in range(len(A)):
        rows.append(i)
    cols = []
    for i in range(len(A[0])):
        cols.append(i)
    n = min(len(A),len(A[0]))  
    while n>0:        
        new_rows =  list(combinations(rows,n))#包含了行所有的可能
        new_cols = list(combinations(cols,n))#包含了列所有的可能
        B = zeros(n,n)
        #print(new_rows)
        #print(new_cols)
        for a in range(len(new_rows)):
            for b in range(len(new_cols)):       
                for i in range(n):
                    for j in range(n):
                        B[i][j] = A[new_rows[a][i]][new_cols[b][j]]   
                #print_matrix(B,0)
                #print("det:",det(B))
                #print()
                #print()
                if det(B) != 0:
                    return len(B) 
                    #pass            
        n-=1
        if n==0:
            return 0
# modifies the original input matrix!
def approximate_matrix(A,appr = 10):
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = round(A[i][j],appr)
    return A
### 有bug，问题在于python除法不太精确，所以需要自己写一个    
def row_echelon(M):
    # 写一个模拟人工化简矩阵的函数，可能超长
    #用递归试一下 
    # row,col 用于定位非零首元
    A = deepcopy(M)
    row = 0
    col = 0
    
    while row<len(A) and col<len(A[0]):
        if is_nonzero(A[row][col]):
            for i in range(row+1,len(A)):
                #add_rows_by_factor_approximate(A,i,-A[i][col]/A[row][col],row) # 这里精度限制在10位，计算 -A[i][col]/A[row][col] 时
                                                                               # 除法精度达不到要求 相加会消不掉，所以取一个能保证消掉的精度
                add_rows_by_factor(A,i,-A[i][col]/A[row][col],row)
                #print((row,col),i,"added:")
                #print_matrix(A,40)

            row += 1 #加到最后会溢出，不满足行列判断条件，跳出循环
            col += 1 #

        else:
            if below_all_zero(A,row,col):
                col+=1 #如果这个元素之下全都是0，就开始看下一列
            else:
                
                i,j = row,col
                while is_zero(A[i][j]) and i<len(A)-1:#
                    i+=1
                exchange_rows(A,row,i)

    return approximate_matrix(A,8)  # 精度上还是有问题，折衷之计


def row_echelon_display_process(M):
    # 写一个模拟人工化简矩阵的函数，可能超长
    #用递归试一下 
    # row,col 用于定位非零首元
    A = deepcopy(M)
    row = 0
    col = 0
    print("original matrix")
    print_matrix(A)
    while row<len(A) and col<len(A[0]):
        if is_nonzero(A[row][col]):
            for i in range(row+1,len(A)):
                #add_rows_by_factor_approximate(A,i,-A[i][col]/A[row][col],row) # 这里精度限制在10位，计算 -A[i][col]/A[row][col] 时
                k =  -A[i][col]/A[row][col]                                   # 除法精度达不到要求 相加会消不掉，所以取一个能保证消掉的精度
                add_rows_by_factor(A,i,k,row)
                if round(k,4) != 0:
                    print('r',i+1,"+(",round(k,4),")r",row+1)
                    print_matrix(A)

            row += 1 #加到最后会溢出，不满足行列判断条件，跳出循环
            col += 1 #

        else:
            if below_all_zero(A,row,col):
                col+=1 #如果这个元素之下全都是0，就开始看下一列
            else:
                
                i,j = row,col
                while is_zero(A[i][j]) and i<len(A)-1:#
                    i+=1
                exchange_rows(A,row,i)
                print("exchange r",row+1,"and r",i+1)
                print_matrix(A)
    return approximate_matrix(A,6)  # 精度上还是有问题，折衷之计

# 接收row_echelon的矩阵，找到pivot的位置，返回几个下标的tuple，在求rref时用得到
def find_pivot(A):

    '''
    print(A)
    print(row_echelon(A))
    print(A==row_echelon(A))
    if row_echelon(A)!=A:
        raise ValueError("input is not a Row Echelon!")
    '''
    #print_matrix(A)
    index_list = []
    for i in range(len(A)-1,0-1,-1):
        for j in range(len(A[0])):
            if A[i][j] == 0:
                continue
            else:
                index_list.append((i,j))
                break
    return index_list
def rref(A):
    B = row_echelon(A)
    index_list = find_pivot(B)
    for i in range(len(index_list)):
        multiply_row(B,index_list[i][0],1/B[index_list[i][0]][index_list[i][1]]) # 这里偶尔会出现除以0的错误，可以用随机阵暴力测试，需要继续研究原因
        for row in range(index_list[i][0]):
            #print(row)
            add_rows_by_factor(B,row,-B[row][index_list[i][0]],index_list[i][0])
            #print(-B[row][index_list[i][0]])
    return approximate_matrix(B,10)


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

    return solve_linear_equation(B,b) 


# 区分无解（返回False），唯一解（返回解向量），无穷解（返回基础解系）
def solve_lineq_homo(A,print_ans = False):#这个暂时是齐次的，require一个系数矩阵
    #先判断解的类型
    if rank(A)<min(len(A),len(A[0])):   ##？？？如果是行数大于列数的情况呢？满秩也不过小于行数，这样一定是无穷解吗？
        # 有无穷多个解
        A = row_echelon(A)
        boundary = find_pivot(A)[0] # 最下面一个pivot，右边都是自由变量
        print(boundary)
        ans = nulls(min(len(A),len(A[0]))-rank(A)) # 基础解系的个数n-r(A)
        k=0
        for j in range(boundary[1]+1,len(A[0])):
            for i in range(len(A)):
                if i < boundary[0]+1:
                    ans[k].append([-A[i][j]])
                else:
                    if i == j:
                        ans[k].append([1])
                    else:
                        ans[k].append([0])
            k+=1
        if print_ans == True:
            for n in range(len(ans)):
                print_vector(ans[n],2)
        return tuple(ans)
        # 想想怎么依次取1，其余取0
        ###未完待续#####
    else:
        return zeros(len(A),1)


# 点积，自适应行向量and列向量
def dot(a,b,appr = 10):
    if matrix_shape(a)!=matrix_shape(b):
        raise ValueError("inputs should be VECTORS of SAME TYPE!")
    elif len(a)!= 1 and len(a[0])!= 1:
        raise ValueError("inputs should be VECTORS!")
    if len(a) == 1: # 说明是行向量
        s = 0
        for j in range(len(a[0])):
            s+=a[0][j]*b[0][j]
        return round(s,appr)
    if len(a[0])==1: # 说明是列向量
        s = 0
        for i in range(len(a)):
            s+=a[i][0]*b[i][0]
        return round(s,appr)
def matrix_add(A,B):
    if matrix_shape(A)!=matrix_shape(B):
        raise ValueError("inputs should be in SAME SHAPE!")
    C = zeros(len(A),len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j]+ B[i][j]
    return C
def matrix_minus(A,B):
    if matrix_shape(A)!=matrix_shape(B):
        raise ValueError("inputs should be in SAME SHAPE!")
    C = zeros(len(A),len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j]- B[i][j]
    return C
def times_const(k,A):
    C = zeros(len(A),len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = k*A[i][j]
    return C
def norm(vec):
    if len(vec) == 1: # 说明是行向量
        s = 0
        for j in range(len(vec[0])):
            s+=vec[0][j]**2           
        return sqrt(s)
    if len(vec[0])==1: # 说明是列向量
        s = 0
        for i in range(len(vec)):
            s+=vec[i][0]**2
        return sqrt(s)
def unitize(vec):
    return times_const(1/norm(vec),vec)
# Gram Schmidt Orthogonalization 接收的是矩阵，一般来讲，向量组中向量的个数应等于向量的维数   
def schmidt_matrix(A):  #理论上来讲应该接收向量组
    if len(A)>len(A[0]):
        raise ValueError("num of vectors should be at least equal to the dimension of vectors! (make sure cols>=rows)")
    a = split_col(A)   # col_vec_group
    b =  zeros(len(A[0]),1) # orth_vec_group
    
    for i in range(len(a)):
        s = zeros(len(a),1)
        if i == 0:
            b[0] = a[0]
            b[0] = times_const(1/norm(b[0]),b[0])

        else:
            for j in range(0,i):
                s = matrix_add(times_const(dot(b[j],a[i])/dot(b[j],b[j]),b[j]),s) # 还得写一个矩阵对应元素相加
            b[i] = matrix_minus(a[i],s)             # 正交化
            #print_vector(b[i])
            #b[i] = times_const(1/norm(b[i]),b[i])   # 单位化
            b[i] = unitize(b[i])       # 单位化
    return tuple(b) # 返回向量组

# 剥一层tuple皮，然后把列向量转换成矩阵, 功能等同于comb_col, 但传入的是tuple, 传入一堆可变参数和传入一个tuple，效果是一样的
def grouptuple_2_matrix(vec_tuple):
    M = zeros(len(vec_tuple[0]),len(vec_tuple))
    for i in range(len(vec_tuple[0])):
        for j in range(len(vec_tuple)):
            M[i][j] = vec_tuple[j][i][0]          
    return M
def schmidt_vec_group(*vec):
    return schmidt_matrix(grouptuple_2_matrix(vec))
def is_lindependent(*vec):
    A = grouptuple_2_matrix(vec)
    if rank(A)<len(vec):
        return True
    else:
        return False
    #print_matrix(grouptuple_2_matrix(vec))
#自适应输入向量组和矩阵
def schmidt(*vecs_or_A):
    
    if len(vecs_or_A)==1:
        A = vecs_or_A[0]
        return schmidt_matrix(A)
    else:
        vecs = vecs_or_A
        return schmidt_matrix(grouptuple_2_matrix(vecs))

def is_orthogonal(A):
    vecs = split_col(A)
    #print_vector_group(vecs)
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            if i!=j:
                if round(dot(vecs[i],vecs[j]),6)!=0: # 无理数精度原因误判，暴力测试出错率十万分之一
                    return False
    return True


def eigen_value(A):
    if check_matrix_square(A):
        import sympy # 不得已，用了sympy作符号计算，为解特征值
        λ = sympy.symbols("λ")
        E = eyes(len(A))
        S = matrix_add(A,times_const(-λ,E))
        #print(sympy.simplify(det(S)))
        return sympy.solve(det(S),λ)

    else:
        raise ValueError("input has to be SQUARE")
def eigen_polynomial(A):
    if check_matrix_square(A):
        import sympy # 不得已，用了sympy作符号计算，为解特征值
        λ = sympy.symbols("λ")
        E = eyes(len(A))
        S = matrix_add(A,times_const(-λ,E))
        return (sympy.factor(sympy.simplify(det(S))))
        

    else:
        raise ValueError("input has to be SQUARE")
def eigen_vector(A):

    val = eigen_value(A)
    E = eyes(len(A))
    for i in range(len(val)):
        S = matrix_add(A,times_const(-val[i],E))
        ##############这里回代矩阵已经有了，就差一个返回解向量的函数
        print(S)

#这两个draw函数效果不尽人意，再改进
def draw_vector_2d(vec):
    import matplotlib.pyplot as plt
    x = [0,vec[0][0]]
    y = [0,vec[0][1]]
    plt.plot(x,y,linestyle='-')

def draw_arrow(vec):
    import matplotlib.pyplot as plt

    ax = plt.plot()
    ax.arrow(0, 0, vec[0][0], vec[0][1],
             width=0.01,
             length_includes_head=True, # 增加的长度包含箭头部分
              head_width=0.25,
              head_length=1,
             fc='r',
             ec='b')
    plt.show()

def show_figure():
	
	import matplotlib.pyplot as plt
	plt.show()

########## Complex Analysis#############
def exp2trig(z):
    print(round(ln(z)))

##########Numerical Analysis#############
#Ch01
#Ch02
#Ch03
#Ch04
#Ch05
#Ch06
## A = LU decomposition
'''
import numpy as np
from scipy.linalg import lu
A = np.array([0,0,1,2,0,0,3,0,1,-1,0,1,2,0,-1,3]).reshape(4,4)
a,b,c= lu(A)
print(a)
print(b)
print(c)
'''

# A = LU decomposition
def lu(A): # 仅限于各阶主子式都不为零，否则需要使用plu分解
    # A 必须是方阵
    L = eyes(len(A))
    U = zeros(len(A),len(A))
    # 函数里面定义函数的操作，还真没见过有用的
    
    def sum_u(L,U,k,j):
        s = 0
        for m in range(k):
            s += L[k][m]*U[m][j]
        return s
    def sum_l(L,U,k,i):
        s = 0
        for m in range(k):
            s += L[i][m]*U[m][k]
        return s
    
    for k in range(len(A)):
        # 求U的第k行 (实际上是k+1)
        for j in range(k,len(A)):
            U[k][j] = (A[k][j] - sum_u(L,U,k,j)) / L[k][k] # / L[k][k] 可删，为保持与U的对称性
            #U[2][j] = (A[2][j] - L[2][0]*U[0][j] - L[2][1]*U[1][j]) / L[2][2] 

        # 求L的第k列 (实际上是k+1)
        for i in range(k,len(A)):
            #L[i][2] = (A[i][2] - L[i][0]*U[0][2] - L[i][1]*U[1][2]) / U[2][2]
            L[i][k] = (A[i][k] - sum_l(L,U,k,i)) / U[k][k]
    return L,U

# A = PLU decomposition
#失败，PA之后还是有0主元的情况，不知问题出在那里##
def plu(A): #
    print("Waring: This algorithm is not well implemented")
    P = eyes(len(A))    
    for j in range(len(A)):
        max_index = j
        max_item = 0
        print(j,'round')
        for i in range(j,len(A)):
            if abs(max_item) < abs(A[i][j]): # 列选主元，其实不必最大，只要主元！=0即可，不知是不是这里的问题
                max_item = A[i][j]
                max_index = i
        if max_index != j:
            A = exchange_rows(A,j,max_index)
            P = exchange_rows(P,j,max_index)
            print("exchanged",j,'and',max_index)
            for k in range(j+1,len(A)):
                A = add_rows_by_factor(A,k,-A[k][j]/A[j][j],j)
                print('消掉：',k)
                print_matrix(A)
    print_matrix(P,name = "P")
    L,U = lu(multiply(P,A))
    return transpose(P),L,U


# Cholesky decomposition
def cholesky(A, mode = 'LLT'):
    L,U = lu(A)
    #print_matrix(L,name = "L = ")
    #print_matrix(U,name = "U = ")
    L_T = U
    D = zeros(len(A),len(A))
    for i in range(len(A)):
        D[i][i] = U[i][i]
        for j in range(i,len(A)):
            U[i][j]/=U[i][i]
    if mode == 'LDLT':
        return L,D,L_T
    if mode == 'LLT':
        D_sqrt  = D
        for i in range(len(A)):
            D_sqrt[i][i] = sqrt(D_sqrt[i][i])
        return multiply(L,D_sqrt),multiply(D_sqrt,L_T)

#Ch07
#Ch08
#Ch09
#




########## 概率论与数理统计 ###########
# PLAYGROUND. run and play, which is the fun part!
# Tips: row vectors should be written as [[1,2]]; column vectors written as [[1],[2]]
# 约定：矩阵和向量都二维list，向量组是在tuple套着几个二维list，看起来是三维
# 挖一个大坑：把矩阵用类表示，重写整个mathy程序，用类表示的矩阵含有propety，在一些问题上比较好处理
# 写一个把向量画出来的函数,像3B1B那样数形结合

# 做一个GUI，线代工具箱，封装起来，更加实用，受众更广
# 用GUI以后，迭代几个版本，底层改用numpy，外带一些自己写的numpy没有的功能，比如化为row_echelon，判断正交等
if __name__ == '__main__':
    #import matplotlib.pyplot as plt

    # 修改意见：传进去的矩阵返回的时候不要把原矩阵改了，看看指针引用的问题

    # matrix_shape 只是看第一行元素来数，最好写一个check_matrix看看传入的到底是不是
    # 写一个能求齐次方程通解的函数
    #### row_echelon 有bug，偶尔会化不成0，而是一个很小的数
    #### 出现这种情况，两次row——echolon操作都化不成零，进而影响后面的rref操作
    #### 抽时间重写一下算法



    
    
    '''
    ###解线性方程组
    A = [[1,2,-1],
        [2,1,3],
        [-1,4,5]]
    b=[ [1],
        [7],
        [3]]
    #LU分解法
    L,U = lu(A)
    y = solve_linear_equation(L,b)
    print_vector(y)
    x = solve_linear_equation(U,y)
    print_vector(x)
    #增广矩阵法
    print_vector(solve_augmented_mat(comb_col(A,b)))
    '''

    '''
    ### 施密特正交化生成正交矩阵
    P = grouptuple_2_matrix(schmidt_matrix(A)) # 把正交向量组转成正交矩阵
    print(is_orthogonal(P))
    '''

    '''
    ###矩阵的cholesky分解测试
    A = [[1,2,-2],
        [2,5,-3],
        [-2,-3,21]] 
    a,c = cholesky(A,mode = "LLT")
    print_matrix(a,name = "L")
    print_matrix(c,name = "LT")

    print_matrix(multiply(a,c))#multiply(b,c)))


    '''


