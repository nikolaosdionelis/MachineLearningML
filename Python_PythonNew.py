# use: http://interactivepython.org/runestone/static/pythonds/index.html#

# website: http://interactivepython.org/runestone/static/pythonds/index.html#
# we use: http://interactivepython.org/runestone/static/pythonds/BasicDS/toctree.html

# we use lambda expressions in Python
# use: https://docs.python.org/2/reference/expressions.html#lambda

# we use: https://docs.python.org/2/reference/expressions.html
# website: https://docs.python.org/2/reference/expressions.html#lambda

import numpy as np

# we use Python's build-in functions
# use: https://docs.python.org/3/library/functions.html

# we use *args and **kwargs
# https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/

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

# we now use *args and **kwargs
# use: https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/

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

# **kwargs must be at the end and hence after *args
def apodosi(*apodoseis, **kwargs):
    # we use the "max" key in the dictionary
    if "max" in kwargs:
        n = kwargs["max"]
    else:
        n = len(apodoseis)

    k = 1
    for i in range(n):
        k *= apodoseis[i]

    return k

# **kwargs must be at the end and hence after *args
def apodosi2(*apodoseis, **kwargs):
    # we use the "max" key in the dictionary
    if "max" in kwargs:
        # we use min(., len(apodoseis))
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
print('')

m = [2.3, 1.4, 1.8, 1.5, 2.4]

# we use: "*m" amd "myFunction(*m)"

# when we have a list m, then we use "*m" to get its elements
print(apodosi(*m, max=3))
print(apodosi2(*m, max=3))

# use *list1 to break the list
print(apodosi2(*m, max=13))
# the function does not work if we do not use "*"

# use *args and **kwargs in functions
# website: https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/

# use: https://www.geeksforgeeks.org/args-kwargs-python/



# convert to binary
# convert the number n to binary
n = 14

# we use the stack data structure
# define a list that will be used as a stack
stack1 = []

# stack: the last item that enters the stack is the first item out
# the stack data structure is Last In First Out (LIFO)
# the queue data structure is First In First Out (FIFO)

print('')

# every program uses an execution stack
# the execution stack in Python is short

# Every program has a stack that contains the parameters and the local variables of the functions
# that have been called. The stack is LIFO. The last parameter of a function gets out first, i.e. LIFO,
# when many funnctions have been called in a recursion.

# recursion problems
# recursion and memoization
# Fibonacci series and memoization

# the stack overflow error
# stack overflow: when recursion, when the execution stack is full

# we use a while loop
while n != 0:
    # d is the last digit
    d = n % 2

    # print(d)
    stack1.insert(0, d)

    # we remove the last digit
    n = n // 2

# print the elements
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

toBinary(14)
print()

# d is the last digit
# d = n % 2
# stack1.insert(0, d)

# we remove the last digit
#n = n // 2

# we use base 8
def toOctal(n):
    if n == 0:
        return
    toOctal(n // 8)
    print(n % 8, end='')

# use base 10
def toDecimal(n):
    if n == 0:
        return
    toDecimal(n // 10)
    print(n % 10, end='')

# 453%10 = 3 = last digit
# 453//10 = 45 = remove last digit

# x%10 = last digit
# x//10 = remove last digit

# we use base 3
def toTernary(n):
    if n == 0:
        return
    toTernary(n // 3)
    print(n % 3, end='')



# sum of N numbers
def sumToN(N):
    sum = 0
    for i in range(1, N + 1):
        sum += i
    return sum

# recursion, sum of N numbers
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

# recursion problems
# coding recursion exercises
# programming recursion exercises

# recursion and memoization
# write code with and without recursion

# use one-line code
# lambda expressions => one line only
# comprehensions, list comprehensions => one line only

# use comprehensions: lists or generator objects

# comprehensions with "(.)" => generator objects
# generator objects are created for one time only

# positional arguments
# define functions and call them with positional arguments
# positional arguments or non-positional arguments, default values

# default values go at the end, *args goes at the end
# use *args and **kwargs, **kwargs goes at the end

# use function1(*list1), use "*list1"
# we use "*list1" to break the list to its elements

# dictionary: keys and values
# dictionaries have keys and values

# we use *args and ** kwargs
# website: https://www.geeksforgeeks.org/args-kwargs-python/

# **kwargs => named arguments, dictionary

# dictionary has keys and values
# we use keys as an index to acccess the values
# "if "max" in dict1:": "max" is a key and not a value

# stack data structure => LIFO
# LIFO, last in first out, stack, execution stack
# recursion, memoization, execution stack, stack overflow

# limited stack, limited short execution stack

# recursion, Fibonacci series => stack overflow
# memoization, we use lookup table, memoization to store values

# Find the n-term of the series: a(n) = a(n-1)*2/3 with recursion and with no recursion.

# recursion for a(n) = a(n-1)*2/3
def function1(n):
    if n == 0:
        return 1
    return (2/3) * function1(n-1)

print('')
print(function1(1))

print(function1(2))
print(function1(3))

print(function1(9))
print('')

# no recursion for a(n) = a(n-1)*2/3
def function2(n):
    k = 1
    for i in range(1,n+1):
        k *= 2/3

    return k

print('')
print(function2(1))

print(function2(2))
print(function2(3))

print(function2(9))
print('')

# Compute the sum 1/2 + 3/5 + 5/8 + .... for N terms with recursion and with no recursion.



# website: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# Machine Learning, AI, Big Data and Data Science
# AI: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# use: http://interactivepython.org/runestone/static/pythonds/index.html

# Data Structures, Algorithms and Data Science
# website: http://interactivepython.org/runestone/static/pythonds/index.html

# List of interview questions:
# www.github.com/MaximAbramchuck/awesome-interview-questions

# The main website for Python coding questions:
# https://www.springboard.com/blog/data-science-interview-questions/#programming

# Python online course: Interactive Python:
# http://interactivepython.org/runestone/static/pythonds/index.html

# Use the book (http://interactivepython.org/runestone/static/pythonds/index.html) as a roadmap.
# Recursion with finite memory and stack. Trees and graphs.

import numpy as np

"""
comprehensions
list comprehensions

A list comprehension is created using [i for i in list1 if i%2 == 0].
The output of a list comprehension is a new list.

The syntax is: result = [transform iteration filter].

filter => filtering condition
The transform occurs for every iteration if the filtering condition is met.
"""

# use: http://interactivepython.org/runestone/static/pythonds/BasicDS/toctree.html
# we now use: http://interactivepython.org/runestone/static/pythonds/index.html

# list comprehensions: one-line code

# The syntax is: result = [transform iteration filter].
# The order in the list matters and the new list has this order.

lista = [1,2,3,4,5,6,7,8,9]
print([i*2 for i in lista])

lista.pop()
print(lista)

lista.pop(0)
print(lista)
print('')

# print all the multiples of 3 and 7 from 1 to 100 using list comprehension
list1 = [i for i in range(1,101) if i%3 == 0 and i%7 == 0]
print(list1)

# 2D array 6x6 with zeros
array2D = [[0 for i in range(0,6)] for j in range(0,6)]
print(array2D)

array2D[0][0] = 1
print(array2D)
print('')

# 3D array 6x6 with zeros
array3D = [[[0 for i in range(6)] for j in range(6)] for k in range(6)]
print(array3D)

print(array3D[0][0][0])
print('')



"""
dictionary

dict = HashMap
dictionaries have keys and values
"""

# =============================================================================
# # Create a function that adds a specific value to the value of a key
# # and if the key does not exist, then create the key.
# =============================================================================

def function1(dict1, key1, value1):
    if key1 not in dict1:
        dict1[key1] = 0

    dict1[key1] += value1
    return dict1

def function2(dict1, key1, value1):
    dict1[key1] = dict1.get(key1, 0) + value1
    return dict1

d1 = {
    'milk': 3.67,
    'butter': 1.95,
    'bread': 1.67,
    'cheese': 4.67
}
print(d1)

d1['butter'] = 2.35
print(d1)
print('')

d2 = {
    1: 3.67,
    2: 1.95,
    3: 1.67,
    4: 4.67
}
print(d2)

d2[2] = 2.35
print(d2)
print('')

d3 = dict([('milk', 3.76), ('butter', 1.95), ('bread', 1.67), ('cheese', 4.67)])
print(d3)

del d3['butter']
print(d3)
print('')

# we use ".format(.)"
print('length of dictionary d3 = {} '.format(len(d3)))

print('length of dictionary d3 = {} compared to {} i in {} '.format(len(d3), d1, d2))
print('')

print(d3.keys())
print(d3.values())

print(d3.items())
print('')

# list1 = dict1.items()
# ".items()" returns a list of tuples

# traverse a dictionary
for food in d1:
    print('{} costs {}'.format(food, d1[food]))

print('')
d1 = function1(d1, 'whine', 4.15)

d1 = function1(d1, 'milk', 1)
print(d1)

d1 = function2(d1, 'whine2', 3.15)
d1 = function2(d1, 'milk', 1)

print(d1)
print('')

# use comprehensions

# use dict comprehension
d4 = {k: v for k, v in enumerate('Good Year John')}
print(d4)

# use: https://docs.python.org/2.3/whatsnew/section-enumerate.html

# we use "enumerate(.)"
# https://docs.python.org/2.3/whatsnew/section-enumerate.html

# website: http://book.pythontips.com/en/latest/enumerate.html

# dict with all letters in "Good Year John"
# without the letters in "John"

d5 = {k: v for k, v in enumerate("Good Year John") if v not in "John"}
print(d5)
print('')

# dict comprehensions => one-line code
# use "enumerate(.)" and dictionary comprehension

# list1 = dict1.keys()
# ".keys()" returns a list

# list2 = dict1.values()
# ".values()" returns a list

# list3 = dict1.items()
# ".items()" returns a list of tuples



"""
Sets
A set has no dublicates.
"""

s = {'a','b','a','c','d'}
print(s)

s2 = set("Good Year John")
print(s2)
print('')

a = set('12345678a')
b = set('1234b')

print('A = ',a)
print('B = ',b)
print('')

a.add('9')
b.remove('4')

print('A = ',a)
print('B = ',b)
print('')

print('A - B = ',a-b) #difference
print('A | B = ',a|b) #Union

print('A & B = ',a&b) #intersection
print('A ^ B = ',a^b) #symmetric difference
print('')

# sets => use Venn diagram
# a Venn diagram solves the problem with sets

# Venn diagram for A-B, A|B, A&B

# AUB is A|B
# AUB is (A OR B)
# AUB is (A Union B)

# A&B is (A Intersection B)
# A^B is (A XOR B)

# XOR = exclusive OR, A XOR B is A^B with sets
# XOR = symmetric difference

# XOR Vs OR: XOR is ^ while OR is |
# OR = | = Union, XOR is exclusive OR

# use: http://mattturck.com/bigdata2018/
# book: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# (1) https://www.jpmorgan.com/global/research/machine-learning
# (2) https://news.efinancialcareers.com/uk-en/285249/machine-learning-and-big-data-j-p-morgan
# (3) https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf



"""
Graphs
Use dict to create graphs.

Graphs are dictionaries in Python.
Dictionaries have keys and values, where the key is the index.

Graphs solve maze problems.
We have directed and undirected graphs.
"""

# traverse a graph
# graphs: binary graphs are a special case of graphs

# maze => graphs
# graphs solve maze problem

# we use a dictionary to create a graph

# graphs are dictionaries
# use dictionaries, lists and sets

# depth first search (dfs)
def dfs(graph, start):
    visited, stack = set(), [start]

    while stack:
        vertex = stack.pop()

        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)

    return visited

# use "extend(.)"
# website: https://www.tutorialspoint.com/python/list_extend.htm

# we use: https://www.tutorialspoint.com/python/list_extend.htm
# use: https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/

# use "yield" instead of "return"
# website: https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/

# do depth first search (dfs)
def dfs_paths(graph, start, goal):
    stack = [(start, [start])]

    while stack:
        (vertex, path) = stack.pop()

        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

# depth first search
# DFS: https://en.wikipedia.org/wiki/Depth-first_search

# we use "next"
# use: https://www.programiz.com/python-programming/methods/built-in/next

# website: https://stackoverflow.com/questions/1733004/python-next-function
# we use: https://www.programiz.com/python-programming/methods/built-in/next

# breadth first search (bfs)
def bfs(graph, start):
    '''
    help bfs
    '''
    visited, queue = set(), [start]

    while queue:
        vertex = queue.pop(0)

        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)

    return visited

# do breadth first search (bfs)
def bfs_paths(graph, start, goal):
    queue = [(start, [start])]

    while queue:
        (vertex, path) = queue.pop(0)

        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

# crate a graph using a dictionary
graph1 = {'A': set(['B', 'C']),
          'B': set(['A', 'D', 'E']),
          'C': set(['A', 'F']),
          'D': set(['D']),
          'E': set(['B', 'F']),
          'F': set(['C', 'E'])}

# hashmap = dict
# dictionaries are hashmaps

# use: help(dict)
# we use: help(dict) and help (list)

# dict: key and value and index = key

print(dfs(graph1, 'A'))
print(list(dfs_paths(graph1, 'C', 'F')))

print('')
print(bfs(graph1, 'A'))

print(list(bfs_paths(graph1, 'C', 'F')))
print(list(bfs_paths(graph1, 'A', 'F')))

# DFS VS BFS
# Graphs: use DFS (or BFS) with or without recursion

# DFS => stack
# BFS => queue

# pandas use dictionaries with list in key and with list in value

# dictionaries have keys and values
# pandas => list in both key and value

# help(dict)

# use: dict(.), set(.), list(.)
# we use: len(dict1)

# from list to dict: dict(list1)
# dict(list1) where list1 has tuples, list1 is a list of tuples

# for OOP, we use classes
# define classes for OOP in Python



import librosa
import soundfile as sf

import numpy as np

magnitude = 0.1
rate = 44100

t = np.linspace(0, 10, rate * 10)

sampling_rate = 16000
audio = magnitude * np.sin(2 * np.pi * 100 * t)

wav_file = 'test_file.wav'
sf.write(wav_file, audio, sampling_rate, subtype='PCM_32')

audio_sf, _ = sf.read(wav_file)
audio_lr, _ = librosa.load(wav_file, sr=None, mono=False)

print('')

#max(np.abs(audio_sf))
print(max(np.abs(audio_sf)))

#max(np.abs(audio_lr))
print(max(np.abs(audio_lr)))
print('')

# we use enumerate(.)
# use: http://book.pythontips.com/en/latest/enumerate.html

#list1 = [4, 5, 1, 2, -4, -3, -5, 0, 0, -5, 1]
list1 = [4, 5, 1, -5, 0, -5]

for counter, value in enumerate(list1):
    print(counter, value)
print('')

for counter, value in enumerate(list1, 1):
    print(counter, value)
print('')

# we use: https://www.geeksforgeeks.org/enumerate-in-python/

list2 = ['apples', 'bananas', 'grapes', 'pears']

counter_list = list(enumerate(list2, 1))
print(counter_list)

# dict(list1) when list1 is a list of tuples
counter_list2 = dict(enumerate(list2, 1))
print(counter_list2)

# dict(list1) or set(list1) or list(set1)
# set(.) => remove the dublicates

print('')
print(set('ABCDABEBF'))
# set has no dublicate entries

# string str is a list of characters
# from list to set, and from string to set

# use: help(list)
# we use: help(list.pop)

# create graphs
#graph1 = {'A' : set(.)}
graph1 = {'A' : set(list1)}

print(graph1)
print('')

# stack => LIFO
# LIFO, stack: extend(.) and pop(.)
# LIFO, stack: append(.) and pop(.)

# queue => FIFO
# FIFO, queue: extend(.) and pop(0)
# FIFO, queue: append(.) and pop(0)

# Depth First Search (DFS) => stack
# Breadth First Search (BFS) => queue

# use "yield"
# yield => return + Generator Object
# Generator Objeccts for better memory, dynamic memory

# "yield" creates a generator object
# generator objects can be only accessed one time (i.e. once)

# we use "exec(.)"
# https://www.geeksforgeeks.org/exec-in-python/

# use: exec(.)
from math import *
exec("print(factorial(5))", {"factorial":factorial})

exec("print(factorial(4))", {"factorial":factorial})
# https://www.programiz.com/python-programming/methods/built-in/exec

# use: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we now use: https://blog.dataiku.com/data-science-trading
# Big Data: https://www.cfasociety.org/cleveland/Lists/Events%20Calendar/Attachments/1045/BIG-Data_AI-JPMmay2017.pdf

# we use: https://www.datasciencecentral.com/profiles/blogs/j-p-morgan-s-comprehensive-guide-on-machine-learning
# https://towardsdatascience.com/https-medium-com-skuttruf-machine-learning-in-finance-algorithmic-trading-on-energy-markets-cb68f7471475

