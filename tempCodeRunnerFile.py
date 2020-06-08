if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    # 待插值的元素值
    x_points = [0,1,2,3,4,5]
    y_points = [1,5,4,8,7,12]
    # 插值
    x = np.linspace(0,5)
    y = list(map(lambda t: newton_interpolate(x_points,y_points,t),x))
    # 画图
    plt.scatter(x_points,y_points)
    plt.plot(x,y)
    plt.show()