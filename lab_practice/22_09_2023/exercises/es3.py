import numpy as np

# Create a random vector of size (3, 6) and find its mean value

a = np.random.randint(0, 100, size=(3, 6))
print(a)

mean = np.mean(a) # First flattens the array, then calculates the mean
print(mean)