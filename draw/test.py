import numpy as np

a = np.mat(np.array([6, 3, 0, 2, 1, -2, 21, -2]))
b = np.array([[1, -1], [1, 3]])
f = np.array([[3, 2, 1], [1, 1, 1], [2, 2, 2]])
d = np.mat(np.array([[2, 3], [1, 2]]))
e = np.array([True, False])
c = np.nonzero(a)
u = d - b

print(np.argsort(a))
