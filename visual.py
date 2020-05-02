############## Visual #############
# no time to implement, try manim

#这两个draw函数效果不尽人意，再改进
def draw_vector_2d(vec):
    import matplotlib.pyplot as plt
    x = [0,vec[0][0]]
    y = [0,vec[0][1]]
    plt.plot(x,y,linestyle='-')

def draw_arrow(vec):
    import matplotlib.pyplot as plt

    ax = plt.plot()
    ax.arrow(0, 0, vec[0][0], vec[0][1],
             width=0.01,
             length_includes_head=True, # 增加的长度包含箭头部分
              head_width=0.25,
              head_length=1,
             fc='r',
             ec='b')
    plt.show()

def show_figure():
	
	import matplotlib.pyplot as plt
	plt.show()
