# --*-- encoding:utf-8 --*--
import pandas as pd
import numpy as np  # 导入一个数据分析用的包“numpy” 命名为 np
import matplotlib.pyplot as plt


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


if __name__ == '__main__':
    dataMat = np.mat([10, 8, 9, 11])
    std = dataMat.std()
    # 设定 x 轴前两个数字是 X 轴的开始和结束，第三个数字表示步长，或者区间的间隔长度
    x = np.arange(4, 14, 0.02)
    # 设定 y 轴，载入刚才的正态分布函数
    y = normfun(x, dataMat.mean(), std)
    plt.plot(x, y)
    plt.grid(True)
    plt.title('Time distribution')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    # 输出
    plt.show()
