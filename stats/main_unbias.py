import random
from playStats.descriptive_stats import mean
from playStats.descriptive_stats import variance
import matplotlib.pyplot as plt



def variance_bias(data):
    n = len(data)
    if n<=1:
        return None
    mean_value = mean(data)
    return sum((e - mean_value)**2 for e in data) / n

def sample(num_of_samples, sample_sz, var):
    data = []
    for _ in range(num_of_samples):
        data.append(var([random.uniform(0.0,1.0) for _ in range(sample_sz)])) #在0-1的均匀分布中抽取sample_sz数量的个体组成一个样本，取该样本均值
    return data

if __name__ == "__main__":
    #biased
    data1= sample(1000, 40, variance_bias)
    plt.hist(data1, bins = "auto", rwidth = 0.8)
    plt.axvline(x=mean(data1), c="black")
    plt.axvline(x=1/12, c="red") #计算0-1均匀分布的总体的方差:（1-0）**2/12
    print("bias: ", mean(data1), 1/12) #打印实验值和理论值
    plt.show()

    #unbiased
    data2 = sample(1000, 40, variance)
    plt.hist(data2, bins="auto", rwidth=0.8)
    plt.axvline(x=mean(data2), c="black")
    plt.axvline(x=1/12, c="red")  # 计算0-1均匀分布的总体的方差:（1-0）**2/12
    print("unbias: ", mean(data2), 1/12)  # 打印实验值和理论值
    plt.show()
