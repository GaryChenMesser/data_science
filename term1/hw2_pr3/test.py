import numpy as np

a = np.ones((10, 2))
b = np.zeros((2, 2))
b = np.zeros((10, 1))
c = np.concatenate((a, b), axis = 1)
print(c)
