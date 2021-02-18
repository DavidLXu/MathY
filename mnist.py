from neural_mat import *

if __name__ == "__main__":
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
    net = Network([784, 128, 10])
    

    # 导入的数据是 (60000,784,) (60000,10,) 需要处理一下维度 -> (60000,784,1) (60000,10,1)
    X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],-1))
    Y_train=Y_train.reshape((Y_train.shape[0],Y_train.shape[1],-1))
    # 导入的数据是 (10000,784,) (10000,10,) 需要处理一下维度 -> (10000,784,1) (10000,10,1)
    X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],-1))
    Y_test=Y_test.reshape((Y_test.shape[0],Y_test.shape[1],-1))

    #训练模型
    #net.train(labels=Y_train[:60000],data_set=X_train[:60000],rate=0.7,epoch=100)
    #net.train(labels=Y_train[:10000],data_set=X_train[:10000],rate=0.7,epoch=5, verbose=1, freq = 1000)

    #保存模型
    #save_modal(net,'net_mnist_matrix_20201112_60000x100_784_64_10_rate0.7.pkl')

    #读取模型
    net = load_modal('net_mnist_matrix_20201112_60000x100_784_128_10_rate0.7acc0.9788.pkl')
    for i in range(1000,1005):
        print("label:",get_result(Y_test[i]),"predict:",get_result(net.predict(X_test[i])))
        plt.imshow(X_test[i].reshape((28,28)))
        plt.show()

    #print("accuracy:",evaluate(network=net, test_data_set=X_test[:10000], test_labels=Y_test[:10000]))
