import numpy as np

a = np.mat(np.array([6, 3, 0, 2, 1, -2, 21, -2]))
b = np.mat([[-3, 1], [1, -3]])
f = np.mat([[0, 0, -2], [1, 2, 1], [1, 0, 3]])
d = np.mat(np.array([[2, 1], [4, 3]]))
dd = np.mat(np.array([[1, 1], [2, 2]])).T
e = np.array([True, False])
c = np.nonzero(a)

t,v=np.linalg.eig(f)
print(t)
print(v)