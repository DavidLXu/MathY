# some numerical algorithms
from linalg import *
from basic import *

########### Numerical Analysis ##########
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

if __name__ == '__main__':
            
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
    A = [[2,1,-4],[1,2,2],[-4,2,20]]
    L,U=lu(A)
    print_matrix(L,name = 'L')
    print_matrix(U,name = 'U')


