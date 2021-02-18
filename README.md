# MathY 1.0.0

Welcome to MathY! MathY is a simple math toolbox for **educational or recreational purpose** only. It is made by a current junior ME student who has wide and wild interests in math and programming. From the very beginning, MathY requires no other third party libraries for basic calculations. Later on, in order to handle complex numerical problems, a bit of sympy is used, which is in a limited case. Overall, MathY is almost  a pure python math solver. It is still under construction and needs your help! Feel free to reach me at xulixincn@163.com .  :)

## How to use?

* Run in terminal: ` python -i mathy-latest.py`.(interactive mode)
* Write your code in `mathy-latest.py`, under `if __name__ == "__main__:" `, then run the script. (script mode)
* For windows users, run `MathY.bat` in MathY directory. (interactive mode)

You can use `dir()` to have a quick look of all supported functions.

## Basic operations

MathY is based on python, which means all pythonic characteristics is supported.  Each entry of calculation is behind three arrows " >>> ". For elementary math learners, basic operations (+ - * / ...) can be used as follows:

```python
>>> 1+1
2
>>> 2-1.2
0.8
>>> 3*2
6
>>> 3/5
0.6

# some constants
>>> pi
3.1415926535897932384626
>>> e
2.718281828459

# some is—functions definition
>>> is_even(3)
False
>>> is_odd(3)
True
>>> is_prime(7)
True
>>> is_decimal(3.1)
True
>>> is_integer(7)
True

# some basic functions
>>> abs(-0.6)  # use Abs() for complex numbers (which is from sympy)
0.6
>>> floor(3.2)
3
>>> ceil(3.4)
4
>>> min(1,2,3) 
1
>>> max(3,2,1)
3
```

Below are some of predefined math functions, mostly elementary functions:

```python
factorial(x)
factorial_2(x)
combination(n,m)

sin(x)
cos(x)
tan(x)

sqrt(x) # binarysearch and newton method both supported
exp(x)
pow(x,y) # where x,y can be real numbers

ln(x)
log10(x)
log(x,y) # logarithm of y on the base of x
"""not implemented yet"""
arcsin(x)
arccos(x)
arctan(x)
```

From the very beginning, MathY is designed not to use any third-party libraries, and these functions are implemented purely with python grammas without even `import math`.

There may be some precision issues, but since MathY is an educational math library, every function is in a clearly-defined way. It's really nice when seeing how the math skyscraper is built using only basic python language properties, and it's a great way to gain a better understanding of mathematics.

## Algebra

Algebra is an important part of math, especially for math beginners, where it is full of variables and unknowns, and it is pretty hard to implement from scratch. If you need to solve symbolic algebra problems, please use sympy.

## Calculus

In `mathy/calculus.py`, multiple hand written differential and integral numerical solvers are provided. Again, for the purpose of not using third-party libraries, symbolic calculus (e.g. indefinite integral) is not supported.

For instance, if you want to numerically find out the derivative of a given function at a given point, you may use:

```python
>>> derivative(sin,0.1)
0.9950041652613538
```

You can change the method by using optional parameters:

```python
>>> derivative(tan,pi/4,method='backward',h=0.05)
1.9062750736999057
```

Or if you want to integrate a function over a certain interval:

```python
>>> integrate(lambda x: x, 0,1)
0.4999999999999995
```

The reason why the result is not 0.5 is because this is merely an approximate solution. For different approaches:

```python
integrate_regular(function,start,end,precision = 2500) # slow, do not recommend
integrate_trapezoid(function,a,b,n)
integrate_simpson(function,a,b,n)
integrate_gauss(function,start,end,n=3)
```

Try it out for different approaches. Again, this is an only educational and recreational math library. For symbolic or professional uses, please use sympy.

## Linear Algebra

Reluctant to admit, though, the reason I wrote this whole thing is because I had a lot of hard-calculated homework when I was taking linear algebra class, and I didn't want to repeat the same method over and over again. :) There are already very professional software like MATLAB, Mathematica, Octave, etc. However, I was interested in how things work behind these softwares. With an inquisitive idea in mind, I began to create the very first version of MathY. 

As a result, `mathy/linalg.py` may be the most verbose and detailed part of the whole program (for now).

In MathY, matrices are represented as 2-layer lists. I know this is a bad idea for large matrix and fast computing (maybe a bad idea to use python to do all these stuff), but when it comes to readability and simplicity, especially for non-CS students who do not have the experience of reading high-level C++ code or complex algorithms, the list-form works the best.

To clearly show the structure of vectors and matrices:

```python
# matrix
A = [[1,2],[3,4]]
B = [[1,2,3],
     [2,3,4],
     [3,4,5]]
# column vector
b = [[1],[2]]
c = [[1],
     [0]]
# row vector
r = [[2,2,1]]
"""
NOTICE: All matrices and vectors in MathY are 2-dimensional
If you try to use a single list to represent vectors,
that may cause serious problems.
When referencing items in vectors, don't forget to use
vec[i][0] for the i-th item in a column vector, and 
vec[0][i] for the i-th item in a row vector, where [0] is
needed to keep the previous agreement.
"""
# vector group
g = [[[0],
      [0],
      [1]],
     [[0],
      [1],
      [0]],
     [[1],
      [0],
      [0]],
    ]
"""
vector group is used when returning multiple vectors, 
for instance Schmidt orthongonalization.
"""
```

DISCLAIMER: Linear Algebra library is only at an elementary level, it may have some bugs, and heavy load calculations is not recommended. (You may try large matrices if you want to give your computer a hard time :)  )

Below are some most useful functions:

```python
# Beautifully print matrices and vectors
print_matrix(A,precision=2,name = 'Matrix')
print_vector(A,precision=4)
print_vectors(*vec_tuple,precision=4)
print_vector_group(vec_tuple,precision=4) # for vector group, where vec_tuple has a dimension 3

# Show matrix property
matrix_shape(A)
is_orthogonal(A)
# TODO is_fullrank(A)
# TODO is_square(A)
# TODO is_diagonal(A)
# TODO is_symmetric(A)
# or combined altogether, print_property

# Generate some specific matrices
zeros(row,col)
ones(row,col)
eyes(n)
diag(*a)
randmat(row,col,largest = 10)

# Concatenate, split
comb_col(*matrices)
comb_row(*matrices)
split_col(A)
split_row(A)


# Matrix elementary transformation
exchange_rows(B,r1,r2) 
exchange_cols(B,c1,c2)
multiply_row(B,r,k)
add_rows(B,r1,r2)
add_rows_by_factor(B,r1,k,r2) # bad precision because of denominator

# Matrix operation
transpose(A)
add_mat(A,B) # TODO: supports multiple inputs mat
sub_mat(A,B)
matrix_add(A,B) # same as add_mat, remains due to historical reasons
matrix_minus(A,B) # same as sub_mat
times_const(k,A) # return kA
multiply(A,B,...) # supports multiple inputs mat
power(A,k)
det(A)
minor(A,row,col) 
adjoint(A)
inv(A)
rank(A)
dot(a,b,appr = 10) # dot product of vectors

# solve linear equation
row_echelon(M)
rref(A)
solve_linear_equation(A,b)
solve_augmented_mat(A)
solve_lineq_homo(A,print_ans = False) 

norm(vec) # the second norm of vector (module, length)
unitize(vec)
schmidt(*vecs_or_A) # orthognalization

# eigenstuffs
eigen_value(A)
eigen_polynomial(A)
eigen_vector(A)
```

For now, basic linear algebra implementations are provided. In `mathy/numeric.py` more advanced numeric methods are provided.

## Statistics



## Numeric

```python
"""Interpolation"""

"""Regression"""

"""Non-linear equations"""
"""Linear eqations"""
lu(A)
plu(A) # failed, needs to rewrite
cholesky(A)
norm_mat(A)
dlu_decompose(A)
spectral_radius(A)
jacobi_iteration(A,b,epochs=20)
gauss_seidel_iteration(A,b,epochs=20)
sor_iteration(A,b,epochs=20)
"""Eigens"""
rayleigh(A,x)
power_iteration(A,x,epochs=20)
inv_power_iteration(A,x,epochs=10,method = "inv") # not yet implemented
```



## Complex Analysis



## Visualization

Visualization is beyond the scope of MathY. You can use matplotlib for plots, or even manim to generate animations. However, MathY does provide some easy-to-use encapsulation for plots, for instance:

```python
def plot(x,y):
    plt.figure()
    plt.plot(x,y)
    plt.show()
def plot_func(function,start,end,steps = 100):
    x = linspace(start,end,steps)
    y = [function(x[i]) for i in range(len(x))]
    plot(x,y)
   
# this makes plotting a function very easy and simple
plot_func(lambda x: sqrt(exp(-x)*sin(x)**2),0,10)
"""
NOTICE:
Since every functions in 
'lambda x: sqrt(exp(-x)*sin(x)**2)'
is implemented from scratch using numerical 
methods, the final result may not be in the desired 
precision. For more precise purposes, use math.sin(),
math.exp(), or numpy.sin() numpy.exp().

"""
```

