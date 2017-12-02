# --*-- encoding:utf-8 --*--
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

X = np.arange(0, 5, 0.5)
Y = 3 - X
plt.plot(X, Y)
xlim(0, 5)
ylim(0, 5)
plt.plot(X, Y)

plt.show()
