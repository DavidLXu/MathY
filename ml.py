from imports import *
import random
def hypotheses(X,theta):
    if(len(theta) == 1):
        theta = transpose(theta)
    return mul(X,theta)

##---cost functions---
def mse_loss(X,y,theta):
    """
    mean squared error
    L = (y - f(x))**2
    X: table of data (m*n)
    y: vector of output (m*1 supervised true value)
    theta: vector of parameters (nx1 col vec, but 1xn row vec also supported)
    """
    inner = pow(sub_mat(hypotheses(X,theta),y),2)
    #printm(inner)
    result = sum_col(inner)[0][0] / (2*len(y))
    return result
def mae_loss(X,y,theta):
    """
    mean absolute error
    L = |y - f(x)|
    """
    pass


#---different ways of gradient descent---
def bgd(X,y,theta,alpha = 0.0001,iters=100):
    """
    Batch gradient descent
    returns theta in col vector format
    """
    cost = []
    if(len(theta) == 1):
        theta = transpose(theta)
    for i in range(iters):
        
        theta = add(theta,times_const(alpha,mul(transpose(X),sub(y,hypotheses(X,theta)))))
        loss = mse_loss(X,y,theta)
        cost.append(loss)
        print(theta,loss)
        if max(theta) > [1e100]:
            raise ValueError("Overshooting! Decrease alpha to avoid it.")
    return theta,cost


def sgd(X,y,theta,alpha = 0.001,iters=100):
    """
    Stochastic gradient descent
    """
    cost = []
    if(len(theta) == 1):
        theta = transpose(theta)
    for k in range(iters):
        for i in range(len(y)):
            for j in range(len(theta)):

                theta[j][0] = theta[j][0]+alpha*(y[i][0]-hypotheses([X[i]],theta)[0][0])*X[i][j]
        loss = mse_loss(X,y,theta)
        cost.append(loss)
        print(theta,loss)
        if max(theta) > [1e100]:
            raise ValueError("Overshooting! Decrease alpha to avoid it.")
    return theta,cost

def mbgd():
    """
    Mini-Batch Gradient Descent
    """
    pass
def unitize(dat):
    #for vec
    pass

def k_means(k,x_data,y_data):
    #print("x_data:",x_data)
    #print("y_data:",y_data)

    # 随机生成k个means
    centriod_x = [random.uniform(min(x_data),max(x_data)) for i in range(k)]
    centroid_y = [random.uniform(min(y_data),max(y_data)) for i in range(k)]
    #print("x_centroid:",centriod_x)
    #print("y_centroid:",centroid_y)
    plt.scatter(centriod_x,centroid_y,color='red')
    # 存放最近的点的标号
    
    plt.show()

    index_coordinate = []
    # 对每一对坐标点
    for i,j in zip(x_data,y_data):
        dist = []
        # 找距离最近的mean值
        for n,m in zip(centriod_x,centroid_y):
            dist.append(sqrt((i-n)**2+(j-m)**2))
        # 记录最近的点是哪一个mean值
        index_coordinate.append([dist.index(min(dist)),i,j])
    a = np.array(index_coordinate)
    
    a = a[np.argsort(a[:,0])]
    print("closest indices | x | y")
    print(a)
    
    print("how to divide this into 3 pieces? according to the first column") #怎么才能按照第一列分三份？")

if __name__ == "__main__":

    # 随机生成三个聚堆的数据
    x = [random.gauss(0,1) for i in range(6)] + [random.gauss(9,1) for i in range(6)] + [random.gauss(5,1) for i in range(6)]
    y = [random.gauss(0,1) for i in range(6)] + [random.gauss(5,1) for i in range(6)] + [random.gauss(9,1) for i in range(6)]
    plt.scatter(x,y)
    k_means(3,x,y)
