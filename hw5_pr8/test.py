import numpy as np
c = 2
a = np.array(range(100)).reshape((10, 10))
print(a[::2 * c, :])
print(a - 1000)

a -= 50
print(np.positive(a))

b = [[1,2], [3,4]]
print(np.mean(b, axis=0))
