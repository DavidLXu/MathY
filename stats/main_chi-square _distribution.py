import matplotlib.pyplot as plt
import numpy as np
import random
from playStats.descriptive_stats import mean
from playStats.descriptive_stats import variance

chi2 = []
for i in range(50000):
    #x = random.random() #返回一个介于左闭右开[0.0, 1.0)区间的浮点数
    x1 = random.normalvariate(0, 1) #返回一个均值是0，方差是1的正态分布随机数
    x2 = random.normalvariate(0, 1)
    x3 = random.normalvariate(0, 1)
    x4 = random.normalvariate(0, 1)
    x5 = random.normalvariate(0, 1)
    x6 = random.normalvariate(0, 1)
    x7 = random.normalvariate(0, 1)
    x8 = random.normalvariate(0, 1)

    chi2.append(x1**2) # 演示一个自由度chi2分布
    #chi2.append(x1**2+x2**2+x3**2+x4**2+x5**2+x6**2+x7**2+x8**2) # 演示多个自由度chi2分布
    #chi2.append(random.normalvariate(0,1)) # 演示正态分布
    #chi2.append(x) #演示uniform分布

print(variance(chi2)) #打印相关卡方分布方差
#plt.figure(num= "不同自由度卡方分布图")
plt.hist(chi2,bins = 30)
plt.show()






'''
x = np.linspace(-5,5,1000)
f = 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)
plt.plot(x,f)
plt.show()
'''