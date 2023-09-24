import numpy as np

# Given the following square array, compute the product of the elements on its diagonal
#  [[1 3 8] [-1 3 0] [-3 9 2]]

def diagonal_product(arr):
    assert len(arr.shape) == 2 and arr.shape[0] == arr.shape[1] # checking if the array has a square shape
    diag = np.diag(arr)
    return np.prod(diag)


arr = np.array([[1, 3, 8], [-1, 3, 0], [-3, 9, 2]])
print('arr:\n', arr, '\n')

print(diagonal_product(arr))