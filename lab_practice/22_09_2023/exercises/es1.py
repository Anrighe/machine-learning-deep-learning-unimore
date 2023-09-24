import numpy as np

# Write a function that takes a 1d numpy array and computes its reverse vector
# (last element becomes the first)

def fun(numpy_array):

    try:
        assert numpy_array.ndim == 1
        print(a.shape)
        return np.flip(numpy_array)
    except AssertionError:
        print('The vector has more than one dimension')


a = np.arange(10)

print(fun(a))