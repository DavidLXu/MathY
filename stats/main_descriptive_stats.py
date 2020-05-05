from collections import Counter
#import playStats.descriptive_stats as stats #只导入了文件夹，作为stats变量
from playStats.descriptive_stats import *
#from playStats.descriptive_stats import frequency
# #from playStats.descriptive_stats import mode
# from playStats.descriptive_stats import median
# from playStats.descriptive_stats import mean
# from playStats.descriptive_stats import rng
# from playStats.descriptive_stats import quartile
# from playStats.descriptive_stats import variance
# from playStats.descriptive_stats import std

if __name__ == "__main__":

    # 测试频数
    # data = [3, 5, 5, 4, 5, 3, 7, 7, 9, 5, 5, 5, 6, 2]
    # counter = Counter(data)
    # print(counter.most_common()) #显示列表中所有元组
    # print(counter.most_common()[0])
    # print(counter.most_common()[0][0]) #显示第一个元组中第一个值（数据）
    # print(counter.most_common()[0][1]) #显示第一个元组中第二个值（该数据对应频次）
    # # print(dir()) #显示加载模块或变量名称

    # 测试频率
    # freq = frequency(data)
    # #freq = stats.frequency(data) #只导入文件夹时使用
    # print(freq)

    #测试众数
    # data = [3, 5, 4]
    # mode_elements, mode_count = mode(data) #添加两个变量接函数ret和count两个返回值
    # #print(mode_elements, mode_count)
    # #更严谨测试
    # if mode_elements:
    #     print(mode_elements, mode_count)
    # else:
    #     print("Mode does not exist")

    # # 测试中位数
    # data = [3, 4, 1,55,99,34,23]
    # print (median(data))

    # 测试均值
    # data = [3, 18, 1,5,56]
    # print (mean(data))

    #测试极差
    data = [1,6,9,8, 5,9]
    print (rng(data))

    # #测试四分位数
    data = [1,3,4,4,5]
    print(quartile(data))
    #a,b,c = quartile(data)
    #print(a,b,c)

    # 测试方差
    # data = [1,2,3,4,5]
    # print(variance(data))
    # #
    # # #测试标准差
    # data = [1,6,13,23, 5,9]
    # print(std(data))