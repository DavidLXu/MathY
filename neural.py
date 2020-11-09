
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


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str 


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)

class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        return self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)


class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node) 
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def loss(self,label,predict):
        return 0.5*np.sum((np.array(label)-np.array(predict))**2)

    def train(self, labels, data_set, rate, epoch):
        start_time = time.time()
        for i in range(epoch):
            print('epoch %d ...' % i)
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)
                loss = self.loss(labels[d], self.predict(data_set[d]))

                now_time = time.time()
                m, s = divmod(now_time-start_time, 60)
                h, m = divmod(m, 60)
                print('epoch %d of %d, sample %d of %d, loss %f, time used %02d:%02d:%02d' % (i, epoch, d,len(data_set), loss, h,m,s)) # show time & loss
                #print('epoch %d of %d, sample %d of %d, loss %f' % (i, epoch, d,len(data_set), loss)) # do not show time (costs more 5s every min)
                #print('epoch %d of %d, sample %d of %d' % (i, epoch, d,len(data_set))) # 不打印loss(因为是每个sample的loss，不稳定)
            #accuracy = evaluate(self,data_set,labels)
            #print("accuracy:",accuracy)
        end_time = time.time()
        m, s = divmod(end_time-start_time, 60)
        h, m = divmod(m, 60)
        print("Training time used %02d:%02d:%02d" % (h, m, s))

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))

    def dump(self):
        for layer in self.layers:
            layer.dump()

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
    #print(test_labels)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        print("label %d, predict %d"%(label,predict))
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
    net = Network([784, 16, 10])
    # net.train(labels=Y_train[:400],data_set=X_train[:400],rate=0.7,epoch=4) 准确率达91%, net = Network([784, 16, 10])
    net.train(labels=Y_train[:1000],data_set=X_train[:1000],rate=1.4,epoch=2)
    save_modal(net,'net_mnist_all.pkl')
    #net = load_modal('net_mnist.pkl')
    print("accuracy:",evaluate(network=net, test_data_set=X_train[:1000], test_labels=Y_train[:1000]))
 
    #net.dump()
    

    