import random
import matplotlib.pyplot as plt

def toss():
    return random.randint(0,1)

if __name__ == "__main__":

#     indices = [] #列表，存储抛硬币次数
#     freq = [] #列表，存储正面朝上的频率
#     for toss_num in range(10,10000,10): #第一次抛10次，第二次抛20次，抛到100000次，以10为步长
#
#         heads = 0
#         for _ in range(toss_num):
#             if toss() == 0:
#                 heads +=1
#         freq.append(heads / toss_num)
#         indices.append(toss_num)
#     plt.figure(1)
#     plt.plot(indices, freq)
#     plt.show()




    # 扔10000次
    head = 0
    freq = []
    x = []
    for i in range(1,10001):
        if random.randint(0,1) == 1:
            head+=1
        if i%10 ==0:
            freq.append(head/i)
            x.append(i)
    plt.figure(2)
    plt.plot(x,freq)
    plt.show()