import numpy as np

a = np.mat(np.array([3, 3, 0, 2, 0, -2, 1, -2])).T
b = np.array([1, 2])
f = np.array([[3, 2,1],[1,1,1],[2,2,2]])
d = np.mat(np.array([[2, 3], [1, 2], [3, 4]]))
e = np.array([True, False])
c = np.nonzero(a)
# d = a[:, 0]
print(np.linalg.det(f))

