import numpy as np

a = np.mat(np.array([3, 3, 0, 2, 0, -2, 1, -2])).T
b = np.array([1555, 25555555])
d = np.mat(np.array([[2, 3], [1, 2], [3, 4]]))
e = np.array([True, False])
c = np.nonzero(a)
# d = a[:, 0]
print(d)
print("--------")
print(d[:-1])
print("--------")
print(np.var(d[:-1]))
print(np.var(d[:-1]))
