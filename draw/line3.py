# --*-- encoding:utf-8 --*--
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

ridgeWeights=mat([[1,2,3,4],[2,5,6,7]])


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()