import numpy as np

# Given two arrays a and b, compute how many time an item of a is higher than the
# corresponding element of b.
#   a: [[1 5 6 8] [2 -3 13 23] [0 -10 -9 7]]
#   b: [[-3 0 8 1] [-20 -9 -1 32] [7 7 7 7]]

a = np.array([[1, 5, 6, 8], [2, -3, 13, 23], [0, -10, -9, 7]])
b = np.array([[-3, 0, 8, 1], [-20, -9, -1, 32], [7, 7, 7, 7]])

assert a.shape == b.shape


tmp = (a > b) # creates a boolean array for the condition
print('tmp:\n', tmp, '\n')

# selects the elements in which the condition is true
print('How many time an item of a is higher than the corresponding element of b?', a[tmp].size) 