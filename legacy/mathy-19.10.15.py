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

###### linear algebra ######
#print matrix in a more readable way
def print_matrix(A,precision=2):
    print("Matrix[")
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
def print_group(*vec_tuple,precision=4):
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
    for i in range(len(A[0])):
        B[r][i] = k*B[r][i]
    return B

def add_rows(B,r1,r2): # adds r2 to r1
    for i in range(len(B[0])):
        B[r1][i] += B[r2][i]
    return B

# r1,r2 start from 1 as in linear algebra
def add_rows_by_factor(B,r1,k,r2): # r1+k*r2 adds k*r2 to r1
    #r1-=1
    #r2-=1
    for i in range(len(B[0])):
        B[r1][i] += k*B[r2][i]
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
    
    
def row_echelon(A):
    # 写一个模拟人工化简矩阵的函数，可能超长
    #用递归试一下 
    # row,col 用于定位非零首元
    #row = 0
    #col = 0
   
    row = 0
    col = 0
    while row<len(A) and col<len(A[0]):
        if is_nonzero(A[row][col]):
            for i in range(row+1,len(A)):
                #print("r%d+(%d)r%d"%(i,-A[i][col]/A[row][col],row)) 把具体操作用自然语言描述出来
                add_rows_by_factor(A,i,-A[i][col]/A[row][col],row)
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
        #print(row,col)      
    return A


def rank_display_process():
    pass # 化row echelon的过程，可以改装一下row_echelon(A)
def rref(A):
    pass
    #rref 是唯一的，有时候非常重要，把它实现出来
    

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

# 点积，自适应行向量and列向量
def dot(a,b):
    if matrix_shape(a)!=matrix_shape(b):
        raise ValueError("inputs should be VECTORS of SAME TYPE!")
    elif len(a)!= 1 and len(a[0])!= 1:
        raise ValueError("inputs should be VECTORS!")
    if len(a) == 1: # 说明是行向量
        s = 0
        for j in range(len(a[0])):
            s+=a[0][j]*b[0][j]
        return s
    if len(a[0])==1: # 说明是列向量
        s = 0
        for i in range(len(a)):
            s+=a[i][0]*b[i][0]
        return s
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
#自适应输入向量组和矩阵
def schmidt(*vecs_or_A):
    
    if len(vecs_or_A)==1:
        A = vecs_or_A[0]
        return schmidt_matrix(A)
    else:
        vecs = vecs_or_A
        return schmidt_matrix(grouptuple_2_matrix(vecs))

# PLAYGROUND. run and play, which is the fun part!
# Tips: row vectors should be written as [[1,2]]; column vectors written as [[1],[2]]
# 约定：矩阵和向量都二维list，向量组是在tuple套着几个二维list，看起来是三维
# 挖一个大坑：把矩阵用类表示，重写整个mathy程序，用类表示的矩阵含有propety，在一些问题上比较好处理
# 写一个把向量画出来的函数,像3B1B那样数形结合
if __name__ == '__main__':
    #import matplotlib.pyplot as plt

    # 修改意见：传进去的矩阵返回的时候不要把原矩阵改了，看看指针引用的问题

    # matrix_shape 只是看第一行元素来数，最好写一个check_matrix看看传入的到底是不是

    
    A = [[1,2,1],[2,1,2],[3,2,1]]
    b = transpose([[1,2,3]])
    
    print(solve_linear_equation(A,b))
    #print_matrix(grouptuple_2_matrix(schmidt(a_1,a_2,a_3)))
    

    
    
