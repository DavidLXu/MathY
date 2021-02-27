#include <iostream>
#include <Eigen/Core>               // 类似numpy的功能
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

// class SigmoidActivator{
// public:
//     Eigen::MatrixXd forward(Eigen::MatrixXd weighted_input){
//         return 1.0 / (1.0 + np.exp(-weighted_input));
//     }
//     Eigen::MatrixXd backward(Eigen::MatrixXd output){
//         return output * (1 - output);
//     }
// };



int main()
{
    Eigen::Matrix<float, 2, 3> matrix_23;
    Eigen::Vector3d v_3d;

    matrix_23<<1,2,3,4,5,6;
    v_3d<<3,2,1;
    cout << 1 +v_3d;
}