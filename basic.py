###### basics mathematics ######
# absolute values

pi = 3.141592653589793238462643383279
π = pi

e = 2.718281828459

def abs(x):
    if x>=0:
        return x 
    else:
        return -x

def floor(x):
    if x>=0:
        return int(x)
    else:
        return int(x)-1
  
def ceil(x):
    if x>=0:
        return int(x)+1
    else:
        return int(x)
'''
use built-in funtions instead!
def round(x):
    if x>=0:
        if x-int(x)>=0.5:
            return int(x)+1
        else:
            return int(x)
    else:
        if int(x)-x>=0.5:
            return int(x)-1
        else:
            return int(x)

def min(*list):
    t = list[0]
    for i in range(len(list)):
        if t>list[i]:
            t = list[i]
    return t
        
def max(*list):
    t = list[0]
    for i in range(len(list)):
        if t<list[i]:
            t = list[i]
    return t
'''   
###some is—functions###

def is_even(n):
    if n%2==0:
        return True
    else:
        return False

def is_odd(n):
    if n%2==1:
        return True
    else:
        return False

def is_prime(n):
    pass

def is_decimal(n):
    if not is_odd(n) and not is_even(n):
        return True
    else:
        return False

def is_integer(n):
    if is_odd(n) or is_even(n):
        return True
    else:
        return False

def is_zero(n):
	if n == 0:
		return True
	else:
		return False
def is_one(n):
	if n == 1:
		return True
	else:
		return False
def is_nonzero(n):
	if n != 0:
		return True
	else:
		return False
def is_positive(n):
	if n > 0:
		return True
	else:
		return False
def is_negative(n):
	if n < 0:
		return True
	else:
		return False
def is_nonnegative(n):
	if n >= 0:
		return True
	else:
		return False
def is_nonpositive(n):
	if n <= 0:
		return True
	else:
		return False
###### some basic functions ######
def factorial(x):
    if x == 0 or x == 1:
        return 1
    else:
        return x*factorial(x-1)
# 隔项阶乘
def factorial_2(x):

    if(x<=1):
        return 1
    else:
        return x*factorial_2(x-2)


def combination(n,m): # order: selet_item,total_item
    if n>m:
        raise ValueError("the first num must be LESS than of EQUAL to the second num")
    return factorial(m)/factorial(n)/factorial(m-n)

#开根号，二分法
def sqrt_binarysearch(x):
    min_val = 0
    max_val = x
    i=0

    while True:
        ans = (max_val + min_val)/2.0
        #print(ans,max_val,min_val)
        if abs(ans**2-x)<1e-12:
            return ans
        if ans**2 > x:
            max_val = ans
        elif ans**2 < x:
            min_val = ans
        


#开根号，牛顿迭代法，更快
def sqrt_newton(x):
    if(x <= 0):
        return 0
    res = x   
    lastres = 0
    while(abs(lastres-res)>1e-8):
        lastres = res
        res = (res + x/res)/2.0

    return res

#暂且把牛顿迭代法作为官方迭代法
def sqrt(x):
    return sqrt_newton(x)
'''
用泰勒展开求根号没有希望，因为后来误差会越来越大
def sqrt(x):
    temp = 1
    fac_odd = 1
    fac_even = 1
    for n in range(1,30):
        #fac_odd = fac_odd*(2*n-1)
        #print("odd",fac_odd)
        #print("even",fac_even)
        #fac_even = fac_even*2*n
        temp = temp + (-1)**(n+1)*(factorial_2(2*n-1)/factorial_2(2*n))*x**n
    return temp
'''
'''
def sqrt(x):

    temp=1
    temp_n = 1
    for i in range(1,20):
        if i == 1:
            temp_n = 1
        else:
            temp_n=temp_n*(2*i-3)
        temp = temp + (-1)**(i+1)*(temp_n)/(2**i)/factorial(i)*(x-1)**i
    return temp
'''
###### trigonomitry functions ######

# Trig functions using Taylor expansions. In reality, they are calculated by CORDIC algorithms.
def sin(x):
    x = x % (2*pi) # transform any value into [0,2pi] to avoid loss in precision in finite Taylor expansions
    temp = 0
    for i in range(1,20):
        temp = temp + ((-1)**(i-1))*(x**(2*i-1))/factorial(2*i-1)
    return temp

def cos(x):
    x = x % (2*pi)
    temp = 1
    for i in range(1,25):
        temp = temp + ((-1)**(i))*(x**(2*i))/factorial(2*i)
    return temp#float("%.3f"%temp)
    
def tan(x):
    return sin(x)/cos(x)

# very big errors! need to find some alternative methods
def arcsin(x):
    # 接近1时误差较大，建议使用分段，Taylor在1处展开
    if abs(x) > 1:
        raise ValueError("math domain error")
    temp=0
    for n in range(900):
        temp+=factorial_2(2*n-1)/factorial_2(2*n)*x**(2*n+1)/(2*n+1)
    return temp

def arccos(x):
    if abs(x) > 1:
        raise ValueError("math domain error")
    temp=pi/2
    for n in range(900):
        temp-=factorial_2(2*n-1)/factorial_2(2*n)*x**(2*n+1)/(2*n+1)
    return temp
    
def arctan(x):
    # taylor expansion only diverge in [-1,1]
    # when come into |x|>1, 0<1/|x|<1
    # arctanx=π/2-arctan(1/x)
    temp = 0
    if abs(x) <= 1:
        for i in range(900):
            temp = temp + (-1)**i*x**(2*i+1)/(2*i+1)
        return temp
    else:
        
        return π/2-arctan(1/x)
 
###### exponential and logarithm
def exp(x):
    
    return (1+x/100000.0)**100000

def ln(x):
    n = 1000000.0 # 数大了能算准，但是画图出现锯齿
    return n * ((x ** (1/n)) - 1)

def log10(x):
    return ln(x)/ln(10)

def log(x,y): # logarithm of y on the base of x

    return ln(y)/ln(x)

def sinh(x):
    return (exp(x)-exp(-x))/2
    
def cosh(x):
    return (exp(x)+exp(-x))/2
    
def tanh(x):
    return sinh(x)/cosh(x)
''' 
实数pow可以用python自带的**实现，故移除
新的pow功能为向量对应项求幂，在机器学习当中大量使用   
def pow(x,y):
    # y 可以是实数
    # 利用x^y=e^(ln(x)*y)
    # not precise because the exp() and ln() implementation is coarse
    if isinstance(y,float):
        return exp(ln(x)*y)
    else:     
        prod = 1
        if y >=0:
            for i in range(y):
                prod *= x
            return prod
        else:
            for i in range(-y):
                prod /= x
            return prod
'''
def linspace(start_val,end_val,steps = 50,ending = "not included"):
	# no matter included or not inculded, steps always equal to the number of points rather than intervals
    ##if 'numpy' in dir() or 'linspace' in dir(): 
    print("NOTICE: You are calling a built-in linspace instead of the numpy's. ",end = '')
    print("This may not be as precise as you think")
    l = []
    i = start_val
    
    if ending == "included":
        interval = (end_val-start_val)/(steps-1)
        while i <= end_val:
            l.append(i)
            i+=interval
        return l
    else:
        interval = (end_val-start_val)/steps
        while i < end_val:
            l.append(i)
            i+=interval
        return l

def mapping(function,x_list):
    y_list = [function(x_list[i]) for i in range(len(x_list))]
    return y_list
    
if __name__ == "__main__":
    #print(log(2,7840))
    print(arccos(0))
