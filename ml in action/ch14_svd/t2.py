import numpy as np

A = np.mat([[1, 1], [2, 2]])

U, S, VT = np.linalg.svd(A)
print(U)
print("---S----")
print(S)
print("---VT----")
print(VT)
