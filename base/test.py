import numpy as np

a = np.array([3, 3])
b = np.array([1, 0])
c = np.multiply(a, b)

for i in zip(b, a):
    print(i)
