import math
import matplotlib.pyplot as plt
import numpy as np


def test01():
    x = np.arange(0, 1, 0.001)
    y1 = [math.pow(a, a) for a in x]
    # plt.ylim((0, 1))
    plt.plot(x, y1, label='x')

    plt.legend(loc='lower right')
    plt.show()


def test02():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    u = np.linspace(0, 4, 1000)
    x, y = np.meshgrid(u, u)
    z = x + y
    ax.contourf(x, y, z, 20)
    plt.show()


if __name__ == '__main__':
    test02()
