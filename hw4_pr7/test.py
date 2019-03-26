import numpy as np

a = np.ones((3, 3))
print(np.matmul(a[0].reshape(3, 1), a[0].reshape(1, 3)))

print(np.matmul(a[0], a[0]))
print(np.dot(a[0], a[0]))

print(np.log(np.exp(5)))
