# some numerical algorithms
from linalg import *
from basic import *
from visual import *
########### Numerical Analysis ##########
'''
本次期末考试的考点
CH01 误差 有效数字 近似计算若干问题
CH02 Lagrange Newton Hermite 差商 差分
CH03 X
CH04 求积公式的代数精度 梯形 抛物线 复化梯形 复化抛物线
CH05 非线性方程 迭代法收敛 牛顿迭代法 弦位法
CH06 高斯消去 列主元高斯消去 LU分解
CH07 向量范数 矩阵范数 谱半径 Jacobi Gauss-Seidel 收敛性判断
CH09 欧拉法 隐式欧拉法 梯形公式  改进欧拉法 2/4阶龙格库塔方法 局部截断误差 稳定性

'''
# CH01
# 计算有效数字位数
def significant_figures(x_appr,x_prec):
    # 参考精确值x_prec计算x_appr的位数
    p = 0 # the order of the approximated number
    n = 0
    # 以下把x_appr化成规格化浮点数，计算阶数 p
    x = x_appr
    while x >=1:
        x /= 10
        p += 1
    while x < 0.1:
        x *= 10
        p -= 1

    d = abs(x_appr-x_prec)
    while True:       
        n += 1
        if d > 0.5 * 10**(p-n):
            break
    return n-1

# CH02
##---Interpolation--
# 拉格朗日插值
def p(x,a):
    s = 0
    for i in range(len(a)):
        s += a[i]*x**i
    return s

def lagrange_interpolate(x_list,y_list,x): # x是自变量
    if len(x_list) != len(y_list):
        raise ValueError("list x and list y is not of equal length!")
    # 系数矩阵
    A = []
    for i in range(len(x_list)):
        A.append([])
        for j in range(len(x_list)):
            A[i].append(pow(x_list[i],j))
    #print_matrix(A)
    b = []
    for i in range(len(x_list)):
        b.append([y_list[i]])
    #print_vector(b)
    # 求得各阶次的系数
    a = lu_solve(A, b) # 用逆的方法，比较费时间，用数值分析方法优化一下
    #print_vector(a)
    a = transpose(a)[0] # change col vec a into 1 dimension
    val = p(x,a)
    print(x,val)
    return val

# 用来显示牛顿差分表 (recursive)
def difference_list(dlist): # Newton
    if len(dlist)>0:
        print(dlist)
        prev,curr = 0,0
        n = []
        for i in dlist:
            curr = i
            n.append(curr - prev)
            prev = i
        n.pop(0)
        difference_list(n)
''' test code       
difference_list([-1000,29,3,6,25,3,6,7,8,6,5,5,44,100000])
'''

def difference_quotient_list(y_list,x_list = []):
    if x_list == []:
        x_list = [i for i in range(len(y_list))]
    print(y_list)
    prev_list = y_list
    dq_list = []
    dq_list.append(prev_list[0])
    for t in range(1,len(y_list)):
        prev,curr = 0,0  
        m = []
        k = -1
        for i in prev_list:
            curr = i
            m.append((curr - prev)/(x_list[k+t]-x_list[k]))
            prev = i
            k+=1
        m.pop(0)     
        prev_list = m
        dq_list.append(prev_list[0])
        print(m)
    return dq_list

def newton_interpolate(x_list,y_list,x):
    coef = difference_quotient_list(y_list,x_list)
    p = coef[0]
    for i in range(1,len(coef)):
        product = 1
        for j in range(i):
            product *= (x - x_list[j] )
        p += coef[i]*product
    return p

def hermite(x0,x1,y0,y1,y0_prime,y1_prime,x):
    alpha0 = lambda x: ((x-x1)/(x0-x1))**2 * (2*(x-x0)/(x1-x0)+1)
    alpha1 = lambda x: ((x-x0)/(x1-x0))**2 * (2*(x-x1)/(x0-x1)+1)
    beta0 = lambda x: ((x-x1)/(x0-x1))**2 * (x-x0)
    beta1 = lambda x: ((x-x0)/(x1-x0))**2 * (x-x1)
    H = alpha0(x)*y0 + alpha1(x)*y1 + beta0(x)*y0_prime + beta1(x)*y1_prime
    return H
    '''    插值函数测试代码
    f = lambda x: hermite(0,1,5,1,1,0,x)
    x = linspace(0,1,ending="included")
    y = mapping(f,x)
    plot(x,y)
    '''
# 分段线性插值
def segmented_linear_interpolate(xlist,ylist,x):

    """
    n = # of intervals, which is derived from len of xlist
    len of xlist is always one item biger than # of intervals
    """
    #we have to make sure that items in xlist is in order
    
    data = dict(zip(xlist,ylist))
    # 按照key排序，也就是xlist
    data = sorted(data.items(),key=lambda item:item[0])
    data = dict(data)
    xlist = list(data.keys())
    ylist = list(data.values())
    n = len(xlist)-1
    if n == 0:
        raise ValueError("n should be greater or equal to 1")
    # print("segmented interpolate, n =",n)
    # 需要把新来的元素判断一下在哪个区间
    i = -1
    for t in xlist:
        if x >= t:
            i += 1
    if i == -1 or i > len(xlist)-1:
        raise ValueError("x should be between %f and %f"%(xlist[0],xlist[-1]))  
    if i == len(xlist)-1:
        return ylist[i]
    return (x-xlist[i+1])/(xlist[i]-xlist[i+1])*ylist[i] + (x-xlist[i])/(xlist[i+1]-xlist[i])*ylist[i+1]

# 分段Hermite插值
def segmented_hermite_interpolate(x_list,y_list,y_prime_list,x):
    """
    n = # of intervals, which is derived from len of xlist
    len of xlist is always one item biger than # of intervals
    """
    #we have to make sure that items in xlist is in order

    # 按照x_list给y_list排序
    data = dict(zip(x_list,y_list))
    data = sorted(data.items(),key=lambda item:item[0])
    data = dict(data)
    xlist = list(data.keys())
    ylist = list(data.values())
    # 按照x_list给y_prime_list排序
    data = dict(zip(x_list,y_prime_list))
    data = sorted(data.items(),key=lambda item:item[0])
    data = dict(data)
    y_prime_list = list(data.values())


    n = len(xlist)-1
    if n == 0:
        raise ValueError("n should be greater or equal to 1")
    # print("segmented interpolate, n =",n)
    # 需要把新来的元素判断一下在哪个区间
    i = -1
    for t in xlist:
        if x >= t:
            i += 1
    
    if i == -1 or i > len(xlist)-1:
        raise ValueError("x should be between %f and %f"%(xlist[0],xlist[-1]))  
    if i == len(xlist)-1:
        return ylist[i]
    
    alpha0 = lambda x: ((x-xlist[i+1])/(xlist[i]-xlist[i+1]))**2 * (2*(x-xlist[i])/(xlist[i+1]-xlist[i])+1)
    alpha1 = lambda x: ((x-xlist[i])/(xlist[i+1]-xlist[i]))**2 * (2*(x-xlist[i+1])/(xlist[i]-xlist[i+1])+1)
    beta0 = lambda x: ((x-xlist[i+1])/(xlist[i]-xlist[i+1]))**2 * (x-xlist[i])
    beta1 = lambda x: ((x-xlist[i])/(xlist[i+1]-xlist[i]))**2 * (x-xlist[i+1])
    H = alpha0(x)*ylist[i] + alpha1(x)*ylist[i+1] + beta0(x)*y_prime_list[i] + beta1(x)*y_prime_list[i+1]
    return H

# CH04
#-------Numerical Integration------
# below, numpy is needed because of historical reasons
import numpy as np
# simpson formula and trapezoid method belongs to newton cotes
def integrate_trapezoid(f,a,b,n):
    h = (b-a)/n # 步长
    xi = np.linspace(a,b,n+1) # n+1 个节点
    Tn = h/2 * ((f(xi[0]))+2*sum(f(xi[1:n]))+f(xi[n]))
    print("使用复化梯形公式求得：Tn = %.7f"%Tn)
    return Tn

def integrate_simpson(f,a,b,n):
    h = (b-a)/(2*n) # 步长
    xi = np.linspace(a,b,2*n+1) # n+1 个节点
    #这里索引公式上有所区别
    Sn = h/3 * (f(xi[0])+4*sum(f(xi[1:2*n:2])) + 2*sum(f(xi[2:2*n-1:2]))  + f(xi[2*n]))
    print(sum(f(xi[1:2*n:2])))
    print(sum(f(xi[2:2*n-1:2])))
    print("使用复化Simpson公式求得：Sn = %.7f"%Sn)
    return Sn

# CH05
#--------Non-Linear Equation-------
# 二分法
def binary_search(f,start,end,epsilon = 1e-6):
    k = 0
    x = (end+start)/2
    while abs(f(x))>=epsilon:
        print(k,x)
        if f(start)*f(x) < 0:
            end = x
        else:
            start = x
        x = (end+start)/2
        k+=1
    return x
'''test code
f = lambda x: x**3+5*x**2-12
print(binary_search(f,1,2))      
'''
def newton_iteration():
    pass

# 双点弦位法，弦截法
def secant_method(f = lambda x: x**3-3*x-1,x0 = 1,x1 = 2,epochs = 50):
    x=[0 for i in range(epochs)]
    x[0]=x0# 你调一下这两个数试试，有两个根，不一定收敛到哪个根，有规律
    x[1]=x1
    res = 0
    # 正常运行是一定会报错的，利用这种机制，返回两种结果
    for k in range(1,epochs):
        try:
            x[k+1]=x[k]-(x[k]-x[k-1])/(f(x[k])-f(x[k-1]))*f(x[k])      
        except ZeroDivisionError: # 收敛之后会发生除零错误, 使用break跳出循环 
            print("converged, breaking loop...") 
            res = x[k]          
            break
        except IndexError: # list assignment index out of range
            print("truncated before convergence...")
            res = x[k]
            break
        print(k,x[k+1])
    return res


def secant_method_single(f = lambda x: x**3-3*x-1,x0 = 1,x1 = 2,epochs = 50):
    x=[0 for i in range(epochs)]
    x[0]=x0# 你调一下这两个数试试，有两个根，不一定收敛到哪个根，有规律
    x[1]=x1
    res = 0
    # 正常运行是一定会报错的，利用这种机制，返回两种结果
    for k in range(1,epochs):
        try:
            x[k+1]=x[k]-(x[k]-x[0])/(f(x[k])-f(x[0]))*f(x[k])      
        except ZeroDivisionError: # 收敛之后会发生除零错误, 使用break跳出循环 
            print("converged, breaking loop...") 
            res = x[k]          
            break
        except IndexError: # list assignment index out of range
            print("truncated before convergence...")
            res = x[k]
            break
        print(k,x[k+1])
    return res
'''
#test code    
f = lambda x: x**3-3*x-1
secant_method(f,0,100)
secant_method_single(f,0,100)
'''

# CH06
# --------- Linear Algebra ----------
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
def principal_minor_check(A):
    n = len(A)
    for i in range(1,n+1):
        D = zeros(i,i)
        for j in range(i):
            for k in range(i):
                D[j][k] = A[j][k]
        #print_matrix(D,name = str(i))
        if det(D) == 0:
            print("各阶顺序主子式不全为0")
            return False
    return True

# A = LU decomposition
def lu(A): # 仅限于各阶主子式都不为零，否则需要使用plu分解
    # A 必须是方阵
    # 检查主子式的代码，非常吃计算量，所以就不检查了，若无法分解会直接报division by zero
    '''
    if principal_minor_check(A) == False:
        #raise ValueError("各阶顺序主子式不全为0, 无法分解。如果继续进行，将产生除0错误")
    '''
    L = eyes(len(A))
    U = zeros(len(A),len(A))
    
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
            U[k][j] = (A[k][j] - sum_u(L,U,k,j))      # / L[k][k]  可删，为保持与U的对称性

        # 求L的第k列 (实际上是k+1)
        for i in range(k,len(A)):
            L[i][k] = (A[i][k] - sum_l(L,U,k,i)) / U[k][k]  # 如果报错，是因为顺序主子式有0   
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

# lu分解法求解矩阵方程
def lu_solve(A,b):
    '''

    在不检查主子式的情况下，求解 1000x1000 的线性方程组花费60s
    求解200x200线性方程组仅花费0.5s左右

    '''
    L,U = lu(A)
    n = len(b)
    y = zeros(n,1)
    x = zeros(n,1)
    
    # 先解 y (nx1矩阵)
    y[0][0] = b[0][0]
    for i in range(1,n):
        s = 0
        for j in range(i): ## 这里的下标问题把我整懵了好一会。
                #课本上 \Sigma_{j=1}{i-1}有i-1项，而这里有i项
                # 但是这里的i是比真实下标小一个的，
            s+=L[i][j]*y[j][0]
        y[i][0] = b[i][0]-s # 为保持列向量所以最后要有一个[0]
    #print_vector(y,name = "y2")
    # 再解 x (nx1矩阵)
    x[n-1][0] = y[n-1][0]/U[n-1][n-1]
    for i in range(n-2,-1,-1):
        s = 0
        for j in range(i+1,n): # 这里下标有n-i-1项，课本是\Sigma_{j=i+1}{n}有n-i项，
                        # 考虑i小1，实际项数和课本一样多 
                        # 注意：以后遇到循环次数与下标相关时，不要从代数的角度比对是不是一样多
                        # 比如之前的i与i-1，实际上是一样多的。
            s+=U[i][j]*x[j][0]
        x[i][0] = (y[i][0]-s)/(U[i][i])
    #print_vector(x,name = "x2")
    return x

# this is much quicker than det(), det_by_expansion(),
# however, precision is a problem
def det_by_lu(A):
    L,U = lu(A)
    prod1 = 1
    prod2 = 1
    for i in range(len(A)):
        prod1*=L[i][i]
        prod2*=U[i][i]
    return prod1*prod2
## 线性方程组的迭代法
# 线性方程组的解法 一般记法，矩阵记法

# decompose A into D-L-U
def dlu_decompose(A):
    D = zeros(len(A),len(A))
    L = zeros(len(A),len(A))
    U = zeros(len(A),len(A))
    for i in range(len(A)):
        D[i][i] = A[i][i]
    for j in range(len(A)):
        for i in range(j+1,len(A)):
            L[i][j] = -A[i][j]
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            U[i][j] = -A[i][j]
    return D,L,U

# CH07
#------Iteration for Linear Equation-----
def norm_mat(A,):
    pass
# 计算谱半径，有助于判断收敛性，用到了sympy 的复数模块
def spectral_radius(A):
    eigens = eigen_value(A) # 如果返回复数，是sympy.I
    print(eigens)
    max_value = 0
    for value in eigens:
        if sympy.Abs(value) > max_value:
            max_value = sympy.Abs(value) # 使用sympy 求模
    return max_value    # 谱半径：特征值模的最大值   

# 传入A,b 计算雅可比迭代，若不收敛则报错
def jacobi_iteration(A,b,epochs=20):
    D,L,U = dlu_decompose(A)
    print_matrix(D,name = "D")
    print_matrix(L,name = "L")
    print_matrix(U,name = "U")
    B1 = multiply(inv(D), add_mat(L, U))
    print_matrix(B1,name = "B1 Jacobi matrix")
    s_radius = spectral_radius(B1)
    print("spectral radius: ",s_radius)
    if s_radius >=1:    # 若没有收敛，报错
        raise ValueError("spectral_radius >= 1, not converging!")
    g1 = multiply(inv(D), b)
    x = zeros(len(A),1)
    for i in range(epochs):
        x = add_mat(multiply(B1,x), g1)
        # print_vector(x)
    return x

def gauss_seidel_iteration(A,b,epochs=20):
    '''
    B2=(D-L)^(-1)*U
    g2=(D-L)^(-1)*b
    x=B2*x+g2
    '''
    D,L,U = dlu_decompose(A)
    print_matrix(D,name = "D")
    print_matrix(L,name = "L")
    print_matrix(U,name = "U")
    B2 = multiply(inv(sub_mat(D,L)),  U)
    print_matrix(B2,name = "B2 Gauss-Seidel matrix")
    s_radius = spectral_radius(B2)
    print("spectral radius: ",s_radius)
    if s_radius >=1:
        raise ValueError("spectral_radius >= 1, not converging!")
    g2 = multiply(inv(sub_mat(D,L)), b)
    x = zeros(len(A),1)
    for i in range(epochs):
        x = add_mat(multiply(B2,x), g2)
        # print_vector(x)
    return x

# 松弛法
def sor_iteration(A,b,w,epochs=20): # gauss_seidel 是一种特例
    pass

#-------CH08 计算特征值----------
# 瑞利商，可以帮助计算特征值
def rayleigh(A,x):
    return multiply(transpose(x),A,x)[0][0]/multiply(transpose(x),x)[0][0]

# 幂迭代法 根据《数值分析萨奥尔中译本》第12章 （课本上稍有不同）
def power_iteration(A,x,epochs=20):
    for i in range(epochs):
        u = times_const(1/norm(x),x) # 我们输入的向量归一化
        x = multiply(A,u) # 输入的任意向量，经过足够多次数乘矩阵，方向接近于特征向量
        lam = multiply(transpose(u),x) # 实际上到这一步，求得是瑞利商
        print("---epoch",i,"---")
        print("Eigenvalue",lam)
        print_vector(u)
    return lam,u # 返回特征值和特征向量
'''
A1 = [[3,-4,3],[-4,6,3],[3,3,1]]
power_iteration(A1,[[1],[1],[1]])

A2 = [[4,2,2],[2,5,1],[2,1,6]]
power_iteration(A2,[[1],[1],[1]])
'''


# 反幂迭代法 
def inv_power_iteration(A,x,epochs=10,method = "inv"):

    for i in range(epochs):
        u = times_const(1/norm(x),x)
        
        lam = multiply(transpose(u),A,u)
        
        x = solve_linear_equation(sub_mat(A,times_const(lam[0][0],eyes(len(A)))),u)
        print("epoch",i,": ",lam)
    return lam


# CH09
#-- Solving differential equation
def euler(f = lambda x,y: y+sin(x),y0 = 1,start = 0,end = 1,h = 0.1):
    """
    solve the equation of
    | y' = f(x,y)
    | y(x0) = y0
    put f(x,y) into f
    """
    n = int((end-start)/h) # 区间个数
    x = [start+i*h for i in range(n+1)] # 生成x_i坐标点
    y = [0 for i in range(n+1)] # 生成全0的y_i列表，等待后续循环修改
    y[0] = y0   # 赋初值
    for i in range(0,n):   # 最后一个值不需要被迭代    
        y[i+1]= y[i]+h*f(x[i],y[i]) # 迭代公式
    return x,y   

def euler_improved(f = lambda x,y: y+sin(x),y0 = 1,start = 0,end = 1,h = 0.1):
    """
    solve the equation of
    | y' = f(x,y)
    | y(x0) = y0
    put f(x,y) into f
    """
    n = int((end-start)/h) # 区间个数
    x = [start+i*h for i in range(n+1)] # 生成x_i坐标点
    y = [0 for i in range(n+1)] # 生成全0的y_i列表，等待后续循环修改
    y[0] = y0   # 赋初值
    for i in range(0,n):       # 最后一个值不需要被迭代  
        y[i+1]= y[i]+h/2*(f(x[i],y[i])+f(x[i+1],y[i]+h*f(x[i],y[i]))) # 迭代公式
    return x,y   

def runge_kutta(f = lambda x,y: -y+x+1 ,y0 = 1,start = 0,end = 1,h = 0.1, order = 2):
    n = int((end-start)/h) # 区间个数
    x = [start+i*h for i in range(n+1)] # 生成x_i坐标点
    y = [0 for i in range(n+1)] # 生成全0的y_i列表，等待后续循环修改
    y[0] = y0   # 赋初值
    if order == 2:
        for i in range(n):  # 最后一个值不需要被迭代  
            K1 = h*f(x[i],y[i])
            K2 = h*f(x[i]+h,y[i]+K1)
            y[i+1] = y[i]+0.5*(K1+K2)
        return x,y
    elif order == 4:
        for i in range(n):  # 最后一个值不需要被迭代  
            K1 = h*f(x[i],y[i])
            K2 = h*f(x[i]+h/2,y[i]+K1/2)
            K3 = h*f(x[i]+h/2,y[i]+K2/2)
            K4 = h*f(x[i]+h,y[i]+K3)
            y[i+1] = y[i]+1/6*(K1+2*K2+2*K3+K4)
        return x,y
    else:
        raise ValueError("阶数请输入2或4！")

if __name__ == '__main__':
    a = sympy.symbols("a")
    A = [[10,-1,4],[-1,7,3],[2,-5,a]]
    b = [[1],[0],[-1]]
    D,L,U = dlu_decompose(A)
    # 计算方法第九章作业第一题 欧拉方法解一阶微分方程
    
    f = lambda x,y: y+sin(x)
    y0 = 1
    a = 0
    b = 1
    h = 0.1
    
    # 第二题 龙格库塔方法
    x,y = runge_kutta(order = 2)
    x1,y1 = runge_kutta(order = 4)

    print("---二阶Runge-Kutta---")
    for i in range(len(x)):
        print("x: %.1f,  y: %f"%(x[i],y[i]))
    print("---四阶Runge-Kutta---")  
    for i in range(len(x1)):
        print("x: %.1f,  y: %f"%(x1[i],y1[i]))
    euler_plot = plt.plot(x,y)
    euler_improved_plot = plt.plot(x1,y1)
    legend = plt.legend(["Runge-Kutta-2","Runge-Kutta-4"],loc='upper left')
    plt.show()
    