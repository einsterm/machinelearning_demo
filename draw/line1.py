# --*-- encoding:utf-8 --*--
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

X1 = np.arange(0, 6, 0.1)
X2_1 = X1 + 3
X2_2 = X1 - 3
X2_3 = 3 - X1
X2_4 = -3 - X1
plt.plot(X1, X2_1)
plt.plot(X1, X2_2)
plt.plot(X1, X2_3)
plt.plot(X1, X2_4)
plt.title("XY=-1,1")
xlim(0, 5)
ylim(0, 5)

plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)

plt.show()
