from ml import *
#---week 1 Andrew Ng--- 
### single variable 
'''
dat = readmat("ex1data1.txt")
x1,y = split_col(dat)
x0 = ones(len(x1),1)

X = comb_col(x0,x1)
theta = [[-3],[1]]
#print(mse_loss(X,y,theta))
#print(hypotheses(X,theta))
epochs = 500
alpha = 0.0001
#--bgd--
theta,cost = sgd(X,y,theta,alpha = alpha,iters = epochs)
plt.figure("Gradient Descent with different methods")
plt.subplot(221)
plt.title("Stochastic Gradient Descent")
plt.scatter(x1,y)
plot_func(lambda x: theta[0][0]+theta[1][0]*x,min(x1)[0],max(x1)[0],steps = 100,hold = True)
plt.subplot(222)
plt.title("SGD cost")
plt.plot([i for i in range(epochs)],cost)
#--sgd--
theta,cost = sgd(X,y,theta,alpha = alpha,iters = epochs)
plt.subplot(223)
plt.title("Batch Gradient Descent")
plt.scatter(x1,y)
plot_func(lambda x: theta[0][0]+theta[1][0]*x,min(x1)[0],max(x1)[0],hold = True)

plt.subplot(224)
plt.title("BGD cost")
plt.plot([i for i in range(epochs)],cost)
plt.show()
'''

### multi variable  
dat = readmat("ex1data2.txt")
x1,x2,y = split_col(dat)
x0 = ones(len(x1),1)
X = comb_col(x0,x1,x2)
printm(X)
printm(y)