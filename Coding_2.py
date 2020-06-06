# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/BasicDS/toctree.html

import numpy as np

# use one-line code
# write as few lines of code as possible

# use comprehensions
a = [i for i in range(2, 100 + 1, 2)]
print(a)

# we use list comprehensions
a = [i for i in range(1, 101) if i % 2 == 0]
print(a)

# create a generator object, use "(.)"
a = (i for i in range(1, 101) if i % 2 == 0)
# the generator object can be used only once

# the generator object can be used one time only
print(list(a))
print('')

# positional arguments => position matters
# we can call function1 using "function1(y=1, x=2)"

# function with positional arguments x, y
def function1(x, y):
    return x - y

# positional arguments: the position matters
print(function1(3, 5))

# named arguments, no matter the order
print(function1(y=3, x=5))

# both positional arguments and named arguments
print(function1(4, y=7))
# in functions, position can matter and can not matter

# positional arguments for function
# positional parameters, function inputs, arguments

print('')
print(max(2,6,9,3))
print(sum([2,6,9,3]))

# functions can have default values

# define a function with default values
def func2(x, y=9, z=1):
    # the default value is for z
    return (x + y) * z
    # If we do not give a value for z, then z=1=(default value)

# we can have default values in functions
# default values go to the end of the arguments

# use: (1) default values, (2) *args, (3) **kwargs
# we use default values, one asterisk (i.e. *) and two asterisks (i.e. **)

# default arguments can be only at the end, even more than one
g = func2(2, 5, 7)
print(g)

print('')
for i in range(5):
    print(i, "-", i ** 2)

# use *args at the end
# we use un-named arguments *args

# (1) *args at the end of the arguments in a function
# (2) default values at the end of the arguments in a function

# *args must be in the end of the arguments
def apodosi(*apodoseis):
    k = 1
    for i in apodoseis:
        k *= i

    return k

# use: (1) *args, and (2) **kwargs
# "**kwargs" is a dictionary dict

# we use keys and values
# "**kwargs" is a dictionary and has keys and values

# **kwargs must be at the end after *args
def apodosi(*apodoseis, **kwargs):
    # we use max
    if "max" in kwargs:
        n = min(kwargs["max"], len(apodoseis))
    else:
        n = len(apodoseis)

    k = 1
    for i in range(n):
        k *= apodoseis[i]

    return k

def apodosi2(*apodoseis, **kwargs):  # to *args prepei na einai to teleftaio sthn seira
    # an dothei timi max, na min ypologizei tis ypoloipes apodoseis
    if "max" in kwargs:
        n = min(kwargs["max"], len(apodoseis))
    else:
        n = len(apodoseis)

    k = 1
    for i in range(n):
        k *= apodoseis[i]

    return k

print('')
print(apodosi(1.11, 1.22, 1.31))

print(apodosi2(1.11, 1.22, 1.31))

m = [2.3, 1.4, 1.8, 1.5, 2.4]

print(apodosi(*m, max=13))  # otan exw hdh tis times se mia lista, tote gia na thn "spasei" vazw mprosta ena *
# alliws tha thn theorisei olokliri, san to proto kai monadiko orisma ths function.



# metatropi arithmou sto dyadiko systima
n = 14

stack1 = []  # mia lista pou tha tin xrisimopoiisoume san stoiva.
# Dhl. to teleftaio pou tha mpei, tha to theoroume proto.  Last In First Out LIFO
# antitheta, to queue einai FIFO

print('')

# ÎÂ¨Î±Î¸Îµ ÏÏÏÎ³ÏÎ±Î¼Î¼Î± Î­ÏÎµÎ¹ Î¼Î¯Î± ÏÏÎ¿Î¯Î²Î± ÏÎ¿Ï ÏÎµÏÎ¹Î­ÏÎµÎ¹ ÏÎ¹Ï ÏÎ±ÏÎ±Î¼Î­ÏÏÎ¿ÏÏ ÎºÎ±Î¹ ÏÎ¹Ï ÏÎ¿ÏÎÏÎ¹ÎºÎ­Ï Î¼ÎµÏÎ±Î²Î»Î·ÏÎ­Ï ÏÏÎ½
# ÏÏÎ½Î±ÏÏÎ®ÏÎµÏÎ½ ÏÎ¿Ï Î­ÏÎ¿ÏÎ½ ÎºÎ»Î·Î¸ÎµÎ¯ Î ÏÎµÎ»ÎµÏÏÎ±Î¯Î± ÏÏÎ½Î¬ÏÏÎ·ÏÎ· ÏÎµÏÎ¼Î±ÏÎ¯Î¶ÎµÎ¹ ÏÏÏÏÎ· LIFO
# Î±Î½ ÎºÎ»Î·Î¸Î¿ÏÎ½ ÏÎ¬ÏÎ± ÏÎ¿Î»Î»Î­Ï ÏÏÎ½Î±ÏÏÎ®ÏÎµÎ¹Ï (ÏÏÎ½Î®Î¸ÏÏ ÏÏÎ·Î½ Î±Î½Î±Î´ÏÎ¿Î¼Î®), ÏÏÏÎµ Î¼ÏÎ¿ÏÎµÎ¯ Î½Î± Î´Î¿ÏÎ¼Îµ ÏÎ¿ Î¼Î®Î½ÏÎ¼Î±
# stack overflow, Î¿ÏÏÏÎµ ÏÎ¿ ÏÏÏÎ³ÏÎ±Î¼Î¼Î± ÏÎµÏÎ¼Î±ÏÎ¯Î¶ÎµÏÏÎ±Î¹.

while n != 0:
    d = n % 2  # d einai to teleftaio psifio
    # print(d)
    stack1.insert(0, d)
    n = n // 2  # kovoume to teleftaio psifio
for i in stack1:
    print(i, end="")
print()

def toBinary(n):
    if n == 0:
        return
    toBinary(n // 2)
    print(n % 2, end='')

toBinary(14)
print()

def sumToN(N):
    sum = 0
    for i in range(1, N + 1):
        sum += i
    return sum


def sumToN_rec(N):
    #print(N)
    if N == 1:
        return 1
    # return 1 + sumToN_rec(N-1)
    return N + sumToN_rec(N - 1)

print('')
print(sumToN_rec(4))

#print(sumToN_rec(40000))
print(sumToN_rec(40))

# na ypologistei o n-ostos oros ths akolouthias a(n) = a(n-1)*2/3
# me anadromiko kai me mi anadromiko tropo

# na ypologistei to athroisma 1/2 + 3/5 + 5/8 + .... gia N orous (anadromika kai mi anadromika)




