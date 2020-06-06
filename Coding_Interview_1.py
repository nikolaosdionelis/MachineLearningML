
import numpy as np
import pandas as ps

#help(list)
# use "help(list)" to read description

list1 = []

# from -100 to 100
#for i in range(-100, 100+1, 2):
#    list1.append(i)

# from -90 to 100
for i in range(-90, 100+1, 2):
    list1.append(i)

print(list1)
print("")

# the first: print(list1[0])
# the last: print(list1[-1])

print(list1[0])
print(list1[-1])
print("")

print(list1[::1])
print("")

print(list1[::2])
print("")

# reverse the list
print(list1[::-1])
print("")

# reverse the list
#print([list1[-1:0:-1], list1[0]])
#print([list1[-1:0:-1], [list1[0]]])
print(list1[-1:0:-1] + [list1[0]])
print("")

print(list1[::-2])
print("")

#help(list)

# Kaggle competitions
# https://www.kaggle.com/nd1511
# Hackathons and Kaggle

list2 = ['Nikolaos', 23]

print(list2[0])
print(list2[1])
print("")

#print(list2[2])
#print("")

list2 = [23, 'hello', 145]

print(list2[0])
print(list2[1])
print(list2[2])
print("")

#print(list2[3])
#print("")

print(list2[1][0])
print(list2[1][1])
print("")

# we use: in range(1, n+1, 1)
# use: in range(start, end+1, step size)

# we use for loop
# for i in range(start, end+1, step size)

# if statements
# if else statements
# for loops

# factorial, !
# 0! = 1, 1! = 1

n = int(input("Give a Number: "))

p = 1

# use: range(1, n+1, 1)
for i in range(1, n+1, 1):
    p = p * i

#print(p)
print(n, "! = ", p)

# use dry run
# dry run the code

# run the code like the interpreter

# range(startValue, endValue+1, stepSize)
# range(1,x+1,1): until x only

# recursion, we use return
# function, def, return

# base case to end recursion

# functions: use def and return
def factorial(x):
    p = 1

    # use: range(StartValue, EndValue+1, StepSize)
    for i in range(1, x+1, 1):
        p = p * i

    print(x, "! = ", p)

factorial(n)

# recursive implementations take more memory and are slower
# recursive implementations are neater

# recursive implementations are more understandable

"""
factorial
n! = 1*2*3*..*n, where 1! = 1, 0! = 1
3! = 1 * 2 * 3 = 6
"""
print("")

# define function for factorial
def factorial(x):
    #print("I am in the factorial function.")
    p = 1

    for i in range(1,x+1,1):
        p = p*i

    print(n,"! = ",p)

# Recursive: The function calls itself

# define function for recursive factorial
def recfactorial(x):
    #print("I am in the recfactorial function.")
    if x == 1:
       return 1
    else:
       return x * recfactorial(x-1)

# main program
#n = int(input("Give me a number "))

factorial(n)

print(n,"! = ", recfactorial(n), ' with recursive')
print("")

print("I am back in the main program.")
print("")

# recursive implementations take more memory and are slower

# if we dry run the recursive implementation,
# then we go downwards searching and upwards completing
# if n=3, then:
# 3*recfactorial(2)         =6
# 2*recfactorial(1)         =2
# 1                         =1

# if we dry run the recursive implementation,
# then we go downwards searching and upwards completing
# if n=5, then:
# 5*recfactorial(4)         =120
# 4*recfactorial(3)         =24
# 3*recfactorial(2)         =6
# 2*recfactorial(1)         =2
# 1                         =1

# recursion for factorial
# factorial: n! = n * (n-1)!, if n > 1 and f(1) = 1
# we use range(StartValue, EndValue+1, StepSize)

# https://www.codecademy.com/nikolaos.dionelis
# Code Academy, Nikolaos Dionelis

# https://www.codecademy.com/nikolaos.dionelis

# Hacker Rank, Nikolaos Dionelis
# https://www.hackerrank.com/nd1511

