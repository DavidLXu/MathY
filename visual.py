import matplotlib.pyplot as plt
from basic import *
############## Visual #############
# no time to implement, try manim

def plot(x,y,hold = False):
    if hold == False:
        plt.figure()
    plt.plot(x,y)
    if hold == False:
        plt.show()

def scatter(x,y):
    plt.figure()
    plt.scatter(x,y)
    plt.show()

# extremely simple and useful!
def plot_func(function,start = 0.1,end = 10,steps = 50):
    x = linspace(start,end,steps)
    #y = [function(x[i]) for i in range(len(x))]
    y = list(map(function,x))
    plot(x,y)

#这两个draw函数效果不尽人意，再改进
def draw_vector_2d(vec):
    x = [0,vec[0][0]]
    y = [0,vec[0][1]]
    plt.plot(x,y,linestyle='-')
    plt.show()

def draw_arrow(vec):
    ax = plt.plot()
    ax.arrow(0, 0, vec[0][0], vec[0][1],
             width=0.01,
             length_includes_head=True, # 增加的长度包含箭头部分
              head_width=0.25,
              head_length=1,
             fc='r',
             ec='b')
    plt.show()

if __name__ == "__main__":

    #plot_func(lambda x: sqrt(exp(-x)*sin(x)**2))
    plot(linspace(0,10),mapping(lambda x:sin(x)**2,linspace(0,10)))