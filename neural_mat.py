import random
from numpy import *
from functools import reduce
from imports import *
import pickle # save and read trained network
import time

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
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )
    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def loss(self,label,predict):
        return 0.5*np.sum((np.array(label)-np.array(predict))**2)

    def train(self, labels, data_set, rate, epoch, verbose = 0, freq = 100):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        verbose: 打印多少
        freq: 当verbose=1时，隔项打印
        '''
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
                        loss = self.loss(labels[d], self.predict(data_set[d]))
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
        self.predict(sample)
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










# one-hot -> integer
def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index
# one-hot
def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    #test_data_set=test_data_set.reshape((test_data_set.shape[0],test_data_set.shape[1]))
    #test_labels=test_labels.reshape((test_labels.shape[0],test_labels.shape[1]))

    for i in range(total):
        label = get_result(test_labels[i])
        #print(network.predict(test_data_set[i]))
        predict = get_result(network.predict(test_data_set[i]))
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

if __name__ == '__main__':
    '''
    # 案例一 使用diy的神经网络预测性别，根据身高和体重

    # 前身高后体重
    x_data=[[180,65],[185,72],[175,70],[160,60],[165,56],[158,53]]

    # 1男0女
    y_label=[[1],[1],[1],[0],[0],[0]]

    # 做一下处理
    x_data=np.array(x_data)/200 

    # 两输入 一输出
    #net_gender = Network([2, 2, 1])
    #net_gender.train(labels=y_label,data_set=x_data,rate=0.5,epoch=5000)
    #save_modal(net_gender,'D:/net')

    net_gender = load_modal('D:/net')

    test_height=185/200
    test_weight=70/200
    y = net_gender.predict([test_height,test_weight])
    print(y)
    '''
    

    # 案例二 mnist
    '''
    超参数的确定-经验公式
    m = sqrt(n+l) + alpha
    m = log(2,n)
    m = sqrt(n*l)
    where:
        m : 隐藏层节点数
        n : 输入层节点数
        l : 输出层节点数
        alpha: 1~10 之间的常数
    '''
    
    '''
    # 生成mnist数据，因为已经保存好了，就直接用numpy读取
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
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    Y_train = np.load("Y_train.npy")
    Y_test = np.load("Y_test.npy")
    net = Network([784, 128,64, 10])
    

    # 导入的数据是 (60000,784,) (60000,10,) 需要处理一下维度 -> (60000,784,1) (60000,10,1)
    X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],-1))
    Y_train=Y_train.reshape((Y_train.shape[0],Y_train.shape[1],-1))
    # 导入的数据是 (10000,784,) (10000,10,) 需要处理一下维度 -> (10000,784,1) (10000,10,1)
    X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],-1))
    Y_test=Y_test.reshape((Y_test.shape[0],Y_test.shape[1],-1))

    net.train(labels=Y_train[:60000],data_set=X_train[:60000],rate=0.7,epoch=10)
    #net.train(labels=Y_train[:10000],data_set=X_train[:10000],rate=0.7,epoch=5, verbose=1, freq = 1000)
    save_modal(net,'net_mnist_matrix_20201112_60000x10_784_128_64_10_rate0.7.pkl')
    #net = load_modal('net_mnist_matrix_20201112_60000x10_rate0.7_epoch10_acc0.9748.pkl')
    print("accuracy:",evaluate(network=net, test_data_set=X_test[:10000], test_labels=Y_test[:10000]))

