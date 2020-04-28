"""
Python Basic Math Libraries
David L. Xu
Version 1.0.0
Adapt Python Version 3.7+
Started From Sept 7, 2019

There are alreay bunch of math libraries using C
But how can we implement it ourselves?
Written Purely in Python from Ground up
To Find things to do In Summer holidays
注重数学方法的实现，而不是各种移位等利用语言特性的操作
"""
### basics
# absolute values
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

### trigonomitry functions
# sin(x) using Taylor expansions
def sin(x):
    pass
    
def cos(x):
    pass
    
def tan(x):
    pass
    
def arcsin(x):
    pass
    
def arccos(x):
    pass
    
def arctan(x):
    pass
    
    
def derivative(f,a,method='central',h=0.0001):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h            
    '''
    if method == 'central':
        return float("%.3f" % ((f(a + h) - f(a - h))/(2*h)))
    elif method == 'forward':
        return float("%.3f" % ((f(a + h) - f(a))/h))
    elif method == 'backward':
        return float("%.3f" % ((f(a) - f(a - h))/h))
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
# run and play
def f(x):
    return x**3
print(derivative(f,2))