# --*-- encoding:utf-8 --*--
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = Axes3D(fig)
X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
X, Y = np.meshgrid(X, Y)
Z = -X - Y - 3

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

theta = np.linspace(-5 * np.pi, 5 * np.pi, 100)
z = np.linspace(-3, 3, 100)
r = z**2 + 4
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
ax.legend()


ax.scatter(2, 3, 5)



plt.show()
