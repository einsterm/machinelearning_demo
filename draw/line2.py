# --*-- encoding:utf-8 --*--
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

X1 = np.arange(0, 5, 0.1)
X2_1 = X1 + 3
X2_2 = X1 - 3
plt.plot(X1, X2_1)
plt.plot(X1, X2_2)
plt.title("XY=-1")
xlim(-8, 8)
ylim(0, 5)

plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)

plt.show()
