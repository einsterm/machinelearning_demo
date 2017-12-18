import numpy as np

A = np.mat([[1, 1], [2, 2]])
U1 = A.T * A
lamda1, vector1 = np.linalg.eig(U1)


U2 = A * A.T
lamda2, vector2 = np.linalg.eig(U2)
print(vector2)