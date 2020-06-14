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