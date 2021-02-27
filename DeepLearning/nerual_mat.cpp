#include <iostream>
#include <Eigen/Core>               // 类似numpy的功能
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;
class SigmoidActivator{
public:
    Eigen::MatrixXd forward(Eigen::MatrixXd weighted_input){
        MatrixXd ones = MatrixXd::Ones();
        return ones / (ones + (-weighted_input).array().exp());
    }
    Eigen::MatrixXd backward(Eigen::MatrixXd output){
        return output * (ones - output);
    }
}

class FullConnectedLayer
{
    FullConnectedLayer(int input_size, int output_size, activator){
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
    }
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

}
    


int main()
{

}