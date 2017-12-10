import numpy as np

a = np.mat(np.array([3, 3, 0, 2, 0, -2, 1, -2])).T
b = np.array([1555, 25555555])
d = np.array([2, 3])
e = np.array([True, False])
c = np.nonzero(a)
d = a[:, 0]
f = np.ones((10, 1))
f[d <= 2] = -1
print(f)
