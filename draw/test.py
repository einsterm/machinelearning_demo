import numpy as np

a = np.mat(np.array([6, 3, 0, 2, 1, -2, 21, -2]))
A = np.mat([[1, 0], [2, -3]])
d = np.mat(np.array([[1, 2], [2, 1]]))
dd = np.mat(np.array([[1, 1], [2, 2]])).T
e = np.array([True, False])
f = np.mat([[0, 0, -2], [1, 2, 1], [1, 0, 3]])
c = np.nonzero(f)
# print(c)
U = np.mat([[.16, .99], [.99, -.16]])
S = np.mat([[3.65, 0], [0, .85]])
V = np.mat([[.58, .81], [-.81, .58]])
# print(np.linalg.svd(A))

# unratedItems = nonzero(dataMat[user, :].A == 0)[1]
arr = []
arr.append(7)
arr.append(71)
arr.append(2)
arr.append(75)

s = sorted(arr)
print(d)
print(d.T)
print(d.I)
print(d * d.I)

# print(np.nonzero(f[2, :]))
# -0.6464   	0.7630
# 0.7630   	0.6464
