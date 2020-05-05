from collections import Counter
from math import sqrt

def frequency(data): #定义一个frequency函数，传入参数data
    """频率"""
    counter = Counter(data) #生成一个counter类对象，调用Counter的构造函数(首字母均大写)，传入参数data
    ret = [] #生成一个空列表，存储函数返回值。
    for point in counter.most_common(): #对counter.most_common中内容进行从大到小遍历。每次取出一个元组，存入point变量。point是datapoint缩写。
        ret.append((point[0], point[1] / len(data))) #元组第一个元素是数据，第二个元素是频率，即频数/数据数量
    return ret

def mode(data):
    """众数"""
    counter = Counter(data)
    if counter.most_common()[0][1] == 1:
        return None, None
    count = counter.most_common()[0][1] #设置变量count，记录众数所对应数值出现次数
    ret = []
    for point in counter.most_common():
         if point[1] == count:
             ret.append(point[0])
         else:
             break
    return ret, count #返回两个值

def median(data):
    """中位数"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    return (sorted_data[n //2 - 1] + sorted_data[n //2]) / 2


def mean(data):
    """均值"""
    return sum(data) / len(data)

def rng(data):
    """极差"""
    return max(data) - min(data)

def quartile(data):
    """四分位数"""
    n = len(data)
    q1, q2, q3 = None, None, None #先初始化为None,n<4时显示
    if n >= 4:
        sorted_data = sorted(data)
        q2 = median(sorted_data)
        if n % 2 == 1:
            q1 = median(sorted_data[:n // 2])
            q3 = median(sorted_data[n // 2 + 1:])
        else:
            q1 = median(sorted_data[:n // 2])
            q3 = median(sorted_data[n // 2:])
        return q1, q2, q3
    #else: #可以不要，直接返回None值
        #print("no quartile found for this data")

def variance(data):
    """方差"""
    n = len(data)
    if n<=1:
        return None
    mean_value = mean(data)
    return sum((e - mean_value)**2 for e in data) / (n - 1)

def std(data):
    """标准差"""
    return sqrt(variance(data))
    # n = len(data)
    # if n <= 1:
    #     return None
    # mean_value = mean(data)
    # return (sum((e - mean_value) ** 2 for e in data) / n - 1) ** 0.5
