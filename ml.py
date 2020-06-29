from imports import *

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
    for vec
if __name__ == "__main__":
    pass