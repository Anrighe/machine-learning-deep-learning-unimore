import numpy as np

# Create and normalize the following matrix (use min-max normalization).
# [[0.35 -0.27 0.56] [0.15 0.65 0.42] [0.73 -0.78 -0.08]]

a = [[0.35, -0.27, 0.56], [0.15, 0.65, 0.42], [0.73, -0.78, -0.08]]

a_min = np.min(a)
a_max = np.max(a)

print(a_min)
print(a_max)

# By using the formula (array - min) / (max - min) 
#  the array gets scaled between 0 and 1

a_normalized = (a - a_min) / (a_max - a_min)
print(a_normalized)

# The previous min and max value will become 0 and 1 respectively