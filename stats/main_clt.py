import random
import matplotlib.pyplot as plt
from playStats.descriptive_stats import mean

def sample(num_of_samples, sample_sz):
    data = []
    for _ in range(num_of_samples):
        data.append(mean([random.uniform(0.0,1.0) for _ in range(sample_sz)])) #在0-1的均匀分布中抽取sample_sz数量的个体组成一个样本，取该样本均值
    return data

if __name__ == "__main__":
    data = sample(1000, 40)
    plt.hist(data, bins = 'auto', rwidth = 0.8)
    plt.axvline(x=mean(data),c = 'red') #呈现该组样本均值的均值所对应的垂直线，
    plt.show()