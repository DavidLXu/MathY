# 下面是一些使用MathY的案例，包含一些计算方法作业题和知乎专栏分享用到的源码
# 注意：严谨的数值计算请使用MATLAB，MathY是一个娱乐性质的数学库
# 如果MathY能在数值分析的算法方面帮助到你，欢迎star一下本项目 ;)

from basic import *
from calculus import *
from linalg import *
from statistics import *
from numeric import *
from saveread import *

"""
# 展示LU分解
A = [[1,1/2],[1/3,1/4]]
print(eigen_value(A))
L,U = lu(A)
print_matrix(L)
print_matrix(U)
print_matrix(multiply(L,U))
"""
###===================
'''
# hermite插值
import numpy as np
f = lambda x: hermite(0,1,0,1,-1,-4,x)
x = np.linspace(0,1)#,121,ending="included")
print(x)
y = list(map(f,x))
plt.scatter([0,1],[0,1],color = "orange")
plt.plot(x,y)
plt.show()
'''
###===================    

'''
# 分段hermite插值
import numpy as np
import matplotlib.pyplot as plt

# 待插值的元素值
x_points = [0,1,2,3,4]   
y_points = [5,4,3,2,1]
y_primes = [0,0,0,0,0]

# 分段hermite插值
f = lambda t: segmented_hermite_interpolate(x_points,y_points,y_primes,t)
x = np.linspace(0,4)
y = list(map(f,x))
# 画图
plt.figure("segmented hermite interpolation")
plt.scatter(x_points,y_points,color = "orange")
plt.plot(x,y)
plt.legend(["segmented hermite interpolation","scattered points"])
plt.show()
'''
###===================
   
'''
# 龙格现象的产生
import numpy as np
import matplotlib.pyplot as plt
f = lambda x: 1/(1+25*x**2)
# 待插值的元素值
x_points = np.linspace(-1,1,11)#,ending = "included")
print(x_points)
y_points = list(map(f,x_points))
# 牛顿插值
x = np.linspace(-1,1)
y = list(map(lambda t: lagrange_interpolate(x_points,y_points,t),x))
# 画图
plt.figure("lagrange interpolation")
plt.scatter(x_points,y_points,color = "orange")
plt.plot(x,y)
plt.legend(["lagrange interpolation","scattered points"])
plt.show()



# 龙格现象的解决-分段线性插值
import numpy as np
import matplotlib.pyplot as plt
f = lambda x: 1/(1+25*x**2)
# 待插值的元素值
x_points = np.linspace(-1,1,11)
y_points = list(map(f,x_points))

# 分段线性插值
fx = lambda t: segmented_linear_interpolate(x_points,y_points,t)
x = np.linspace(-1,1,51)
y = list(map(fx,x))
# 画图
plt.figure("segmented interpolation")
plt.scatter(x_points,y_points,color = "orange")
plt.plot(x,y)
plt.legend(["segmented interpolation","scattered points"])
plt.show()
'''

###===================

'''
# 计算方法第九章作业第一题 欧拉方法解一阶微分方程
f = lambda x,y: y+sin(x)
y0 = 1
a = 0
b = 1
h = 0.1
x,y = euler(f,y0,a,b,h)
x1,y1 = euler_improved(f,y0,a,b,h)
print("---欧拉法---")
for i in range(len(x)):
    print("x: %.1f,  y: %f"%(x[i],y[i]))
print("---改进欧拉法---")  
for i in range(len(x1)):
    print("x: %.1f,  y: %f"%(x1[i],y1[i]))
euler_plot = plt.plot(x,y)
euler_improved_plot = plt.plot(x1,y1)
legend = plt.legend(["Euler","Euler imporved"],loc='upper left')
plt.show()

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

'''
###===================
'''
# 高斯赛德尔迭代法和雅可比迭代法求解线性方程组
A = [[1,0.4,0.4],[0.4,1,0.8],[0.4,0.8,1]]
b = [[1],[2],[3]]

gauss_seidel_iteration(A,b)
jacobi_iteration(A,b) # 因为不收敛而报错
'''
###===================
'''
# 差分表
difference_list([1,0,2,-1,3])
# 差商表
difference_quotient_list([1,0,2,-1,3])
'''
###===================
'''
#24
A = [[1,2,6],[2,5,15],[6,15,46]] 
L,U = lu(A)
print_matrix(L,name = "L")
print_matrix(U,name = "U")
print_matrix(multiply(L,U))


#23
A = [[1,-1,1],[5,-4,3],[2,1,1]]
b = [[-4],[-12],[11]]
print_matrix(solve_linear_equation(A,b))
'''
###===================

'''
# 使用自己实现的函数计算第六题的LU分解        
A = [[2,1,-4],[1,2,2],[-4,2,20]]
L,U=lu(A)
print("A的LU分解")
print_matrix(L,name = 'L', precision=3)
print_matrix(U,name = 'U', precision=3)
b=[[-1],[0],[4]]
y = solve_linear_equation(L,b)
x = solve_linear_equation(U,y)
print("方程组的解为")
print_vector(x)
'''
###===================


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
###===================
'''
# 瑞利商和雅可比迭代
A1 = [[1,2,-4],[1,1,2],[1,1,1]]
b = randmat(3,1)
print(rayleigh(A1,b))
jacobi_iteration(A1,b)

'''
