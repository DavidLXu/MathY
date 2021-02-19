import random
import numpy as np
from functools import reduce

import sys
sys.path.append('../') # 在当前目录下才可以运行

#from imports import *
import pickle # save and read trained network
import time
import matplotlib.pyplot as plt

random.seed(123)
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# 归一化
def normalize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# 平移到零点
def centralize(data):
    mu = np.mean(data, axis=0)
    return (data - mu)




# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, 
                 activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,
            (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))
    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        # 式2
        self.input = input_array
        #print("input:",input_array)
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)
        #print("self.output",self.output.shape)
    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array
    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad



# Sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):
        return output * (1 - output)
# 神经网络类
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        self.layers_num = layers # 存一下每层的神经元个数
        
        
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )
    def predict(self, sample,check_dim=False,show = True):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        # 自动转换维度 (784,) -> (784,1) 不然容易出问题。当输入维度正确时可以去掉这一句以加快速度
        if check_dim == True:
            sample = self.check_dimension(sample,show)

        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def loss(self,label,predict):
        return 0.5*np.sum((np.array(label)-np.array(predict))**2)

    def train(self, labels, data_set, rate = 0.7, epoch=2, verbose = 1, freq = 1000,pause_info = 1,check_dim=True):
        '''
        训练函数
        labels:     样本标签
        data_set:   输入样本
        rate:       学习速率
        epoch:      训练轮数
        verbose:    打印多少
        freq:       当verbose=1时，隔项打印
        pause_info: 暂停显示信息的时间
        '''
        if check_dim == True:
            labels,data_set = self.check_dimensions(labels,data_set)

        print("Starting to train...")
        time.sleep(pause_info)
        start_time = time.time()
        if verbose == 0:
            '''
            训练时只打印epoch和最终时间，非常快
            '''
            for i in range(epoch):
                for d in range(len(data_set)):
                    self.train_one_sample(labels[d], 
                        data_set[d], rate)
                now_time = time.time()
                m, s = divmod(now_time-start_time, 60)
                h, m = divmod(m, 60)
                print('epoch %d of %d, time used %02d:%02d:%02d' % (i, epoch,h, m, s))
            end_time = time.time()
            m, s = divmod(end_time-start_time, 60)
            h, m = divmod(m, 60)
            print("Training time used %02d:%02d:%02d" % (h, m, s))

        elif verbose == 1:
            '''
            训练时只打印epoch，sample loss 和实时时间，因为占用stdout，会慢很多
            '''
            for i in range(epoch):
                for d in range(len(data_set)):
                    self.train_one_sample(labels[d], 
                        data_set[d], rate)
                    if d % freq ==0:
                        loss = self.loss(labels[d], self.predict(data_set[d],check_dim=False))
                        now_time = time.time()
                        m, s = divmod(now_time-start_time, 60)
                        h, m = divmod(m, 60)
                        print('epoch %d of %d, sample %d of %d, loss %f, time used %02d:%02d:%02d' % (i, epoch, d,len(data_set),loss,h,m,s))
            end_time = time.time()
            m, s = divmod(end_time-start_time, 60)
            h, m = divmod(m, 60)
            print("Training time used %02d:%02d:%02d" % (h, m, s))
        else:
            print("please input verbose = 1 or verbose = 0")

    def train_one_sample(self, label, sample, rate):
        '''
        训练一个样本，注意label和sample应该是二维的
        '''
        self.predict(sample,check_dim=False)
        self.calc_gradient(label)
        self.update_weight(rate)
    def calc_gradient(self, label):
        #label=label.reshape((label.shape[0],-1))
        #print(label.shape,self.layers[-1].output.shape)
        #print(self.layers[-1].activator.backward(self.layers[-1].output).shape )
        #print(self.layers[-1].output)
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta
    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def check_dimensions(self, labels, data_set):
        '''
        检查输入数据集和标签集的维度是否和网络相匹配
        如果是简单的维度问题，可以自动转换数据集的维度
        (TODO)如果是网络定义的问题，可以提供输入层和输出层的建议
        '''

        # 防止传进来的不是numpy数组
        data_set = np.array(data_set)
        labels = np.array(labels)

        required_input_shape = (data_set.shape[0],self.layers_num[0],1)
        required_output_shape = (labels.shape[0],self.layers_num[-1],1)
        print(f"Input dimensions: \t dataset {data_set.shape}, \t labels {labels.shape}.")
        print(f"Required dimensions: \t dataset {required_input_shape}, \t labels {required_output_shape}.")

        # 若不符合，自动改变维度（仅限小维度错误比如28x28->784，大维度错误比如数据和网络不匹配也无能为力）
        if data_set.shape != required_input_shape or labels.shape != required_output_shape:
            #之前的版本，不够智能
            #labels = labels.reshape((labels.shape[0],labels.shape[1],-1))
            #data_set = data_set.reshape((data_set.shape[0],data_set.shape[1],-1)) 
            data_set = data_set.reshape(required_input_shape)
            labels = labels.reshape(required_output_shape)
            print(f"Input changed to: \t dataset {data_set.shape}, \t labels {labels.shape} to match neural network.")           
        else:
            print("Dimensions matched!")
        return labels,data_set

    def check_dimension(self,sample,show = True):
        """
        检查单个样本的标签是否满足维度要求
        """
        # 防止传进来的不是numpy数组而是list导致后续无法测量维度
        sample = np.array(sample)
        required_sample_shape = (self.layers_num[0],1)

        if show == True:
            print(f"Input dimensions: \t sample {sample.shape}.")
            print(f"Required dimensions: \t sample {required_sample_shape}.")

        # 若不符合，自动改变维度（仅限小维度错误比如28x28->784，大维度错误比如数据和网络不匹配也无能为力）
        if sample.shape != required_sample_shape:
            sample = sample.reshape(required_sample_shape)
            if show == True:
                print(f"Input changed to: \t sample {sample.shape} to match neural network.")           
        else:
            if show == True:
                print("Dimensions matched!")
        return sample


def get_result(vec):
    '''
    实现数据类型的转换 one-hot -> integer
    相当于 np.argmax()
    自动适应 [1,2,5,4] 和 [[1],[2],[5],[4]] 以及numpy型向量
    '''
    return (list(vec).index(max(vec)))


# one-hot
def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    # 若数据维度不符，自动转换维度
    #test_data_set=test_data_set.reshape((test_data_set.shape[0],test_data_set.shape[1],-1))
    #test_labels=test_labels.reshape((test_labels.shape[0],test_labels.shape[1],-1))
    test_labels,test_data_set=network.check_dimensions(test_labels,test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        #print(network.predict(test_data_set[i]))
        predict = get_result(network.predict(test_data_set[i],check_dim=False))
        #print("label %d, predict %d"%(label,predict))
        if label != predict:
            error += 1
    return 1-((error) / (total))

def save_modal(network,path):
    with open(path,'wb') as pkl_file:
        pickle.dump(network,pkl_file)
        print("Network modal saved successfully as", pkl_file.name)
    
def load_modal(path):
    with open(path,'rb') as pkl_file:
        print("Network modal loaded from",pkl_file.name)
        return pickle.load(pkl_file)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])



###########################
### 一些案例，写在函数里 ###
###########################

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x
def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x

#b = np.array([[1,2,3],[4,5,6]])
#print(Z_ScoreNormalization(b,b.mean(),b.std()))


# 案例一 房价预测
def house_price():
    import pandas as pd
    data = pd.read_csv(r'./DeepLearning/波士顿房价.csv')

    # 分开train和label
    X = data.iloc[:,:13]
    Y = data.iloc[:,13]

    # 归一化
    X_1 = (X-X.min())/(X.max()-X.min())
    Y_1 = (Y-Y.min())/(Y.max()-Y.min())
    
    # 分开训练集和测试集
    pos = 400
    X_train = X_1.iloc[:pos,:]  # .drop("CHAS",axis=1)
    Y_train = Y_1.iloc[:pos]
    X_test = X_1.iloc[pos:,:]
    Y_test = Y_1.iloc[pos:]
    
    # 转换维度
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    
    net = Network([13,13,1])
    net.train(labels=Y_train,data_set=X_train,epoch=1000,rate = 0.7)

    #X_test = X_test.reshape(X_test.shape[0],13,1)
    for i in range(50):
        # 注意 predict 要求的维度 
        pred_1 = net.predict(X_test[i],check_dim=True,show = False) # 自动检查并转换维度，而且不显示信息
        label_1 = Y_test[i]

        pred = (Y.max()-Y.min())*pred_1+Y.min()
        label = (Y.max()-Y.min())*label_1+Y.min()

        print("predict: %.3f, \tlabel: %.3f"%(pred[0][0],label))


























'''
# 生成mnist数据，因为已经保存好了，就直接用numpy读取
# 保存成npy文件的时候，已经归一化了
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test)=mnist.load_data()

X_train = X_train[:].reshape((-1,784))
X_test = X_test.reshape((-1,784))
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train[:],10)
Y_test = np_utils.to_categorical(y_test,10)

np.save("X_train",X_train)
np.save("X_test",X_test)
np.save("Y_train",Y_train)
np.save("Y_test",Y_test)
'''
# 案例二 
def mnist():

    dir_path = "mnist-dataset/"
    X_train = np.load(dir_path+"X_train.npy")
    X_test = np.load(dir_path+"X_test.npy")
    Y_train = np.load(dir_path+"Y_train.npy")
    Y_test = np.load(dir_path+"Y_test.npy")

    # 导入的数据是 (60000,784,) (60000,10,) 需要处理一下维度 -> (60000,784,1) (60000,10,1)
    #_train=X_train.reshape((X_train.shape[0],784,-1))
    #Y_train=Y_train.reshape((Y_train.shape[0],Y_train.shape[1],-1))
    # 导入的数据是 (10000,784,) (10000,10,) 需要处理一下维度 -> (10000,784,1) (10000,10,1)
    #X_test=X_test.reshape((X_test.shape[0],784,-1))
    #Y_test=Y_test.reshape((Y_test.shape[0],Y_test.shape[1],-1))

    net = Network([784, 10, 10])
    #net.check_dimensions(labels=Y_train,data_set=X_train)
    #net.train(labels=Y_train[:60000],data_set=X_train[:60000],rate=0.7,epoch=4) #准确率达91%, net = Network([784, 16, 10])
    net.train(labels=Y_train[:60000],data_set=X_train[:60000],rate=0.7,epoch=2,check_dim=True) # check_dim=True 自动处理维度问题
    #save_modal(net,'net_mnist.pkl')
    #net = load_modal('net_mnist_all.pkl')
    print("accuracy:",evaluate(network=net, test_data_set=X_test[:10000], test_labels=Y_test[:10000]))


    
    for i in range(1000,1005):
        print("label:",get_result(Y_test[i]),"predict:",get_result(net.predict(X_test[i],check_dim=True))) # 打开check_dim, 自动处理维度问题
        plt.imshow(X_test[i].reshape((28,28)))
        plt.show()
    



'''
## 读取原始 fashion mnist 的代码 后续保存成numpy格式方便使用
## 保存的时候还没有归一化
import gzip    
def read_data():
    files = [
    'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    # 我在当前的目录下创建文件夹，里面放入上面的四个压缩文件
    current = './fashion_mnist_data'
    paths = []
    for i in range(len(files)):
        paths.append('./fashion_mnist_data/'+ files[i])
    
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
        
    return (x_train, y_train), (x_test, y_test)

(train_images, train_labels), (test_images, test_labels) = read_data()

train_labels = get_one_hot(train_labels,10)
test_labels = get_one_hot(test_labels,10)

np.save("train_images",train_images)
np.save("train_labels",train_labels)
np.save("test_images",test_images)
np.save("test_labels",test_labels)
'''
# 案例三 fashion mnist 
def fashion_mnist():  

    dir_path = "fashion-dataset/"
    train_images = np.load(dir_path+"train_images.npy")
    train_labels = np.load(dir_path+"train_labels.npy")
    test_images = np.load(dir_path+"test_images.npy")
    test_labels = np.load(dir_path+"test_labels.npy")

    
    train_images = train_images/255.0
    test_images = test_images/255.0
 
    # 导入的数据是 (60000,28,28) (60000,10,1) 需要处理一下维度 -> (60000,784,1) (60000,10,1)
    #train_images=train_images.reshape((train_images.shape[0],784,-1))
    #train_labels=train_labels.reshape((train_labels.shape[0],train_labels.shape[1],-1))
    # 导入的数据是 (10000,28,28) (10000,10,1) 需要处理一下维度 -> (10000,784,1) (10000,10,1)
    #test_images=test_images.reshape((test_images.shape[0],784,-1))
    #test_labels=test_labels.reshape((test_labels.shape[0],test_labels.shape[1],-1))


    net = Network([784, 10, 10])
    net.train(labels=train_labels[:60000],data_set=train_images[:60000],rate=0.5,epoch=3,verbose=1,freq=500)
    #net.train(labels=Y_train[:10000],data_set=X_train[:10000],rate=0.7,epoch=5, verbose=1, freq = 1000)
    #save_modal(net,'net_fashion_matrix_20201112_60000x10_784_128_128_10_rate0.5.pkl')
    #net = load_modal('net_mnist_matrix_20201112_60000x100_784_128_10_rate0.7acc0.9788.pkl')

    fashion_list = {0:	"T-shirt/top T恤",
                    1:	"Trouser 裤子",
                    2:	"Pullover 套衫",
                    3:	"Dress 裙子",
                    4:	"Coat 外套",
                    5:	"Sandal 凉鞋",
                    6:	"Shirt 汗衫", 
                    7:	"Sneaker 运动鞋",
                    8:	"Bag 包",
                    9:	"Ankle boot 踝靴"}
    for i in range(1000,1005):
        print("label:",fashion_list[get_result(test_labels[i])],"predict:",fashion_list[get_result(net.predict(test_images[i],check_dim=True))])
        plt.imshow(test_images[i].reshape((28,28)))
        plt.show()

    print("accuracy:",evaluate(network=net, test_data_set=test_images[:10000], test_labels=test_labels[:10000]))

# 案例四 与或门
def gate():

    and_data = np.array([[[0],[0]],[[0],[1]],[[1],[0]],[[1],[1]]])
    and_label = np.array([[[0]],[[0]],[[0]],[[1]]])

    or_data=[[0,0],[0,1],[1,0],[1,1]]
    or_label=[[0],[1],[1],[1]]

    net_and = Network([2, 1])
    net_and.train(labels=and_label,data_set=and_data,rate=0.5,epoch=3000)
    net_or = Network([2, 1])
    net_or.train(labels=or_label,data_set=or_data,rate=0.5,epoch=3000,pause_info=0)

    print(net_and.predict([0,1],check_dim=True))
    print(net_or.predict([0,1],check_dim=True))


'''
最早的版本 net.train 在接受数据的时候，维度为三维数组，转换成numpy格式，例如：
and_data = np.array([[[0],[0]],[[0],[1]],[[1],[0]],[[1],[1]]])
and_label = np.array([[[0]],[[0]],[[0]],[[1]]])
现在已经具备了自我调整维度功能，直接输入列表型数据即可
'''
if __name__ == '__main__':

    # 三个案例，打开注释以查看    
    # fashion_mnist()
    # gate()
    # mnist()
    house_price()
    
    # 一个简单的训练加法器的案例
    # data = [[0,0],[0,1],[1,0],[1,1]]
    # label = [[0,1],[1,0],[1,1],[0,0]]
    # net = Network([2,2,2])
    # net.train(labels=label,data_set=data,epoch=3000)
    # print(net.predict([0,0],check_dim=True))

    
    
