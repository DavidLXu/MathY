import matplotlib.pyplot as plt
import random
from collections import Counter

if __name__ == "__main__":
    # scatter plot
    #random.seed(666)  # 随机数种子，可以生成与教程相同的图
    # x = [random.randint(0, 100) for _ in range(150)]  # _ 是个占位符，循环中不关心具体的元素内容，就是简单的“让它循环这么多次”
    # y = [random.randint(0, 100) for _ in range(150)] #生成150个随机数，是1-100间的整数
    # plt.scatter(x, y)
    # plt.show()

    # # line plot
    # x = [random.randint(0, 100) for _ in range(10)]
    # plt.plot ([i for i in range(10)], x)
    # plt.show()

    # bar plot
    # data = [1,4,4,3,4,5,5,3,1,4,4,2,2,4,5,3,4,2,3,3] #可理解为教育程度等级分布
    # counter = Counter(data)
    # x = [point[0] for point in counter.most_common()]
    # y = [point[1] for point in counter.most_common()]
    # plt.bar(x,y)
    # plt.show()

    # histogram
    #random.seed(2020)
    # data = [random.randint(1,100) for _ in range(100)]
    # #plt.hist(data)
    # #plt.hist(data, rwidth = 0.8
    # #plt.hist(data, rwidth=0.8, bins =5) #区间（柱形）数量
    # plt.hist(data, rwidth=0.8, bins =5, density = True) #变为频率直方图
    # plt.show()

    #条形图为分类变量服务，直方图为数值变量服务；
    #直方图中可以对x轴进行任意宽度的区间分割，条形图不可以（因其无等距属性）。

    #boxplot
    # data = [random.randint(1, 100) for _ in range(100)]
    # #data.append(200) #加入一个极端值
    # #data.append(-100)
    # plt.boxplot(data)
    # plt.show()

    ##并排箱线图
    data1 = [random.randint(66, 166) for _ in range(200)]
    data2 = [random.randint(60, 120) for _ in range(200)]
    #plt.boxplot(data1, data2) #两个拼接，暂不清楚意义
    plt.boxplot([data1, data2])
    plt.show()