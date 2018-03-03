import math
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.05, 10, 0.05)
y1 = [math.log(a, 1.5) for a in x]
y2 = [math.log(a, 2) for a in x]
y3 = [math.log(a, 3) for a in x]
plt.plot(x, y1, label='log1.5(x)')
plt.plot(x, y2, label='log2(x)')
plt.plot(x, y3, label='log3(x)')
plt.legend(loc='lower right')
plt.show()
