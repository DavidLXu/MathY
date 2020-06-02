"""
I won't pretend like this is best practice, by in practice, it can be very nice 
to simply have all of the functions and constants of MathY available without having 
to worry about what namespace they come from.

Rather than having a large pile of "from <module> import *" at the top of every such
script, the intent of this file is to make it so that one can just include
"from mathy.imports import *".  The effects of adding more modules
or refactoring the library on current or older scene scripts should be entirely
addressible by changing this file.

Note: One should NOT import from this file for main library code, it is meant only
as a convenience.
"""

from basic import *
from calculus import *
from linalg import *
from statistics import *
from numeric import *
from saveread import *
from info import *
