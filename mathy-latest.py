# -*- coding: UTF-8 -*-
"""
Python Basic Math Libraries
David L. Xu
Version 1.0.0
Adapt Python Version 3.7+
Started From July 7, 2019
First uploaded on my Github acount on Oct 15, 2019
There are alreay bunch of math libraries using C
But how can we implement it ourselves?
Written Purely in Python from Ground up
For recreational use in summer holidays

NOTICE: This library is only for entertainment and 
does not ensure the precision to be tolerated

Simply use basic operations(+-*/) and some language
properties to achieve most of the functions;
Math is magic!

Tips:
1.In Python2, divide(/) two numbers to get an integer
2.To avoid ambiquity, please run the script in Python3
"""
from imports import *


    

if __name__ == '__main__':
    a = 1
    x,y = sympy.symbols('x y')
    A = mat("1 2 3 ;4 5 5 ;3 2 1")
    printm(A)
























   

    '''
    how to make a MathY intepreter?
    import ast
    import copy
    def convertExpr2Expression(Expr):
            Expr.lineno = 0
            Expr.col_offset = 0
            result = ast.Expression(Expr.value, lineno=0, col_offset = 0)

            return result
    def exec_with_return(code):
        code_ast = ast.parse(code)

        init_ast = copy.deepcopy(code_ast)
        init_ast.body = code_ast.body[:-1]

        last_ast = copy.deepcopy(code_ast)
        last_ast.body = code_ast.body[-1:]

        exec(compile(init_ast, "<ast>", "exec"), globals())
        if type(last_ast.body[0]) == ast.Expr:
            return eval(compile(convertExpr2Expression(last_ast.body[0]), "<ast>", "eval"),globals())
        else:
            exec(compile(last_ast, "<ast>", "exec"),globals())

    while True:
        code = input('>> ')
        try:
            exec_with_return(code)
        except BaseException as err:
            print(err)
    '''