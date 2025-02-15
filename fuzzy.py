# fuzzy mathematics
# 致敬扎德祖师爷
from linalg import matrix_shape, zeros

# fuzzy matrix multiplication
def fuzzymm(A,B):
    # 模糊矩阵合成运算(模糊矩阵相乘)的Matlab实现移植
    # 运算规则，先"取小后取大"
    # 输入必须为二阶矩阵A为m行n列, B为n行p列;
    m,n=matrix_shape(A)
    q,p=matrix_shape(B) # 获得输入矩阵的维度信息
    if n!=q:
        raise ValueError('第一个矩阵的列数和第二个矩阵的行数不相同！')
    else:
        R=zeros(m,p) # 初始化矩阵
        for k in range(m):    
            for j in range(p):
                temp=[]
                for i in range(n):
                    Min = min(A[k][i],B[i][j]) #求出第i对的最小值
                    temp.append(Min) #将求出的最小值加入的数组中              
                R[k][j]=max(temp)
    return R
    
if __name__ == "__main__":

    A = [[1,2],[3,4]]
    B = [[5,6],[7,8]]
    print(fuzzymm(A,B))










'''
模糊数学的争议：
1.  理论基础薄弱：模糊度的定义规则 不就是概率得到的数值结果嘛。
    但是又没法明确证明，因为就是想要绕开复杂的概率计算啊
2.  相关体系不完备：过程中的运算及结果本来是带变量的，被模糊数值化了，
    就难保用的定理性质仍然有效，而且很难使用不同于概率思想体系的"模糊"
    思想体系解释清楚，至少现在不行

作者：柳奇
链接：https://www.zhihu.com/question/34898054/answer/75484300
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

Rudy Kalman（一个系统论专家）：“从技术的观点看，扎德教授的理论是需要严厉、
甚至残酷的批判的。这真的很不合适。问题在于，扎德教授到底是提出了一个重要的
理论还是仅仅是他个人的一厢情愿？
William Kahan（一个计算机学家）：“模糊集合论是错的！错的！非常有害的！我不认
为有经典逻辑不能解决的问题。扎德的观点好像是说‘技术将我们带入混乱并且我们无法
走出’。但实际上技术并没有带我们进入混乱，贪婪、软弱和矛盾才是罪魁祸首。我们需
要的是更强的逻辑观点，而不是弱化的。模糊集合论的危险就在于，它会助长带给我们巨
大麻烦的不精确想法的气焰。“

作者：Insinuate
链接：https://www.zhihu.com/question/34898054/answer/110418985
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

'''