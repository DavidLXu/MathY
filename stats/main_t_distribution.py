import matplotlib.pyplot as plt
import numpy as np
import random
from playStats.descriptive_stats import mean
from playStats.descriptive_stats import variance


t = []
for _ in range(100):
    #x = random.random()
    x1 = random.normalvariate(0, 1)
    x2 = random.normalvariate(0, 1)
    x3 = random.normalvariate(0, 1)
    x4 = random.normalvariate(0, 1)
    x5 = random.normalvariate(0, 1)
    x6 = random.normalvariate(0, 1)
    x7 = random.normalvariate(0, 1)
    x8 = random.normalvariate(0, 1)

    chi2 = x1**2+x2**2+x3**2+x4**2+x5**2
    #chi2 = x1**2+x2**2+x3**2+x4**2+x5**2+x6**2+x7**2+x8**2
    t.append(x5/np.sqrt(chi2/5))


plt.figure(num="t分布图")
plt.hist(t,bins = 10) #画出上述特定自由度的t分布图

x = np.linspace(-5,5,1000) #在指定的间隔内返回1000个均匀间隔的数字
f = 1/np.sqrt(2*np.pi)*np.exp(-x**2/2) #标准正态分布图形
plt.plot(x,f)

plt.show()