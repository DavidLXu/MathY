
# 仿照《概率论与数理统计》的讲解思路，待完成。
########## 概率论与数理统计 ###########


from imports import *


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

if __name__ == "__main__":
    x = [random.randint(0, 100) for _ in range(150)]
    a = median(x)
    print(a)