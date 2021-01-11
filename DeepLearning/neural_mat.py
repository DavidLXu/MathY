import random
from numpy import *
from functools import reduce

#import sys
#sys.path.append('../') # 在当前目录下才可以运行

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

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

if __name__ == '__main__':

    
    # 案例二 mnist.py
    """
    # 案例三 fashion mnist
    '''
    ## 读取原始 fashion mnist 的代码 后续保存成numpy格式方便使用
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

    train_images = np.load("train_images.npy")
    train_labels = np.load("train_labels.npy")
    test_images = np.load("test_images.npy")
    test_labels = np.load("test_labels.npy")

    
    train_images = train_images/255.0
    test_images = test_images/255.0
 
    # 导入的数据是 (60000,784,) (60000,10,) 需要处理一下维度 -> (60000,784,1) (60000,10,1)
    train_images=train_images.reshape((train_images.shape[0],784,-1))
    train_labels=train_labels.reshape((train_labels.shape[0],train_labels.shape[1],-1))
    # 导入的数据是 (10000,784,) (10000,10,) 需要处理一下维度 -> (10000,784,1) (10000,10,1)
    test_images=test_images.reshape((test_images.shape[0],784,-1))
    test_labels=test_labels.reshape((test_labels.shape[0],test_labels.shape[1],-1))



    net = Network([784, 128, 128, 10])
    net.train(labels=train_labels[:60000],data_set=train_images[:60000],rate=0.5,epoch=3,verbose=1,freq=5000)
    #net.train(labels=Y_train[:10000],data_set=X_train[:10000],rate=0.7,epoch=5, verbose=1, freq = 1000)
    save_modal(net,'net_fashion_matrix_20201112_60000x10_784_128_128_10_rate0.5.pkl')
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
    for i in range(1000,1015):
        print("label:",fashion_list[get_result(test_labels[i])],"predict:",fashion_list[get_result(net.predict(test_images[i]))])
        plt.imshow(test_images[i].reshape((28,28)))
        plt.show()

    print("accuracy:",evaluate(network=net, test_data_set=test_images[:10000], test_labels=test_labels[:10000]))

    

    # 案例四 cifar10
    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f,encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
            Y = np.array(Y)
            return X, Y

    def load_CIFAR10(ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)#使变成行向量
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    (train_images, train_labels), (test_images, test_labels) = load_CIFAR10()
    