# 以前充斥在程序当中的乱七八糟的注解，就先放到这里吧

# 最先写的是basic.py
# calculus, linalg, statistics 依赖 basics
# complex 依赖 basic, calculus
# numeric 依赖 basics, calculus, linalg

# 添加原则：常用的函数放置在basic




# PLAYGROUND. run and play, which is the fun part!
# Tips: row vectors should be written as [[1,2]]; column vectors written as [[1],[2]]
# 约定：矩阵和向量都二维list，向量组是在tuple套着几个二维list，看起来是三维
# 挖一个大坑：把矩阵用类表示，重写整个mathy程序，用类表示的矩阵含有propety，在一些问题上比较好处理
# 写一个把向量画出来的函数,像3B1B那样数形结合

# 做一个GUI，线代工具箱，封装起来，更加实用，受众更广
# 用GUI以后，迭代几个版本，底层改用numpy，外带一些自己写的numpy没有的功能，比如化为row_echelon，判断正交等


# matrix_shape 只是看第一行元素来数，最好写一个check_matrix看看传入的到底是不是
# 写一个能求齐次方程通解的函数
#### row_echelon 有bug，偶尔会化不成0，而是一个很小的数,影响后面的rref操作
#### 抽时间重写一下算法