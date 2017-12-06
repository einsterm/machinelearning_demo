#!/usr/bin/python
# coding:utf8

import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels


def clip(alpha, L, H):
    ''' 修建alpha的值到L和H之间.
    '''
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha


def select_2(i, m):
    ''' 在m中随机选择除了i之外剩余的数
    '''
    l = list(range(m))
    seq = l[: i] + l[i + 1:]
    return random.choice(seq)


def get_w(alphas, dataset, labels):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    a = labels.reshape(1, -1).T
    yx = labels.reshape(1, -1).T * np.array([1, 1]) * dataset
    print(yx.T)
    w = np.dot(yx.T, alphas)
    return w.tolist()


''' 简化版SMO算法实现，未使用启发式方法对alpha对进行选择.
   :param dataset: 所有特征数据向量
   :param labels: 所有的数据标签
   :param C: 软间隔常数, 0 <= alpha_1 <= C
   :param max_1ter: 外层循环最大迭代次数
   '''


def simple_smo(dataset, labels, C, max_1ter):
    dataset = np.array(dataset)
    m, n = dataset.shape
    labels = np.array(labels)
    # 初始化参数
    chenzi_mat = np.zeros(m)
    b = 0
    it = 0

    "SVM分类器函数 y = w^Tx + b"

    def f(x):
        x = np.matrix(x).T
        data = np.matrix(dataset)
        ks = data * x  # 核函数，求内积
        wx = np.matrix(chenzi_mat * labels) * ks
        fx = wx + b
        return fx[0, 0]

    while it < max_1ter:
        pair_changed = 0
        for i in range(m):
            chenzi_1, data_1, label_1 = chenzi_mat[i], dataset[i], labels[i]  # chenzi_* 为拉格朗日乘子
            f_data_1 = f(data_1)
            E_1 = f_data_1 - label_1
            j = select_2(i, m)
            chenzi_2, data_2, label_2 = chenzi_mat[j], dataset[j], labels[j]
            f_data_2 = f(data_2)
            E_2 = f_data_2 - label_2
            K_11, K_22, K_12 = np.dot(data_1, data_1), np.dot(data_2, data_2), np.dot(data_1, data_2)
            eta = K_11 + K_22 - 2 * K_12
            if eta <= 0:
                print('WARNING  eta <= 0')
                continue
            # 获取更新的alpha对
            chenzi_1_old, chenzi_2_old = chenzi_1, chenzi_2
            chenzi_2_new = chenzi_2_old + label_2 * (E_1 - E_2) / eta
            # 对alpha进行修剪 
            if label_1 != label_2:
                L = max(0, chenzi_2_old - chenzi_1_old)
                H = min(C, C + chenzi_2_old - chenzi_1_old)
            else:
                L = max(0, chenzi_1_old + chenzi_2_old - C)
                H = min(C, chenzi_2_old + chenzi_1_old)
            chenzi_2_new = clip(chenzi_2_new, L, H)
            chenzi_1_new = chenzi_1_old + label_1 * label_2 * (chenzi_2_old - chenzi_2_new)
            if abs(chenzi_2_new - chenzi_2_old) < 0.00001:
                continue
            chenzi_mat[i], chenzi_mat[j] = chenzi_1_new, chenzi_2_new
            # 更新阈值b
            b_1 = -E_1 - label_1 * K_11 * (chenzi_1_new - chenzi_1_old) - label_2 * K_12 * (chenzi_2_new - chenzi_2_old) + b
            b_2 = -E_2 - label_1 * K_12 * (chenzi_1_new - chenzi_1_old) - label_2 * K_22 * (chenzi_2_new - chenzi_2_old) + b
            if 0 < chenzi_1_new < C:
                b = b_1
            elif 0 < chenzi_2_new < C:
                b = b_2
            else:
                b = (b_1 + b_2) / 2
            pair_changed += 1
            # print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(it, i, pair_changed))
        if pair_changed == 0:
            it += 1
        else:
            it = 0
            # print('iteration number: {}'.format(it))
    return chenzi_mat, b


if '__main__' == __name__:
    # 加载训练数据
    dataset, labels = load_data('testSet.txt')
    # 使用简化版SMO算法优化SVM
    chenzi_mat, b = simple_smo(dataset, labels, 0.6, 40)
    # 分类数据点
    points = {'+1': [], '-1': []}
    for point, label in zip(dataset, labels):
        if label == 1.0:
            points['+1'].append(point)
        else:
            points['-1'].append(point)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制数据点
    for label, pts in points.items():
        pts = np.array(pts)
        if label == '+1':
            ax.scatter(pts[:, 0], pts[:, 1], label=label, c='red', )
        else:
            ax.scatter(pts[:, 0], pts[:, 1], label=label, c='blue',)
            # 绘制分割线
    w = get_w(chenzi_mat, dataset, labels)
    x1, _ = max(dataset, key=lambda x: x[0])
    x2, _ = min(dataset, key=lambda x: x[0])
    a1, a2 = w
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    ax.plot([x1, x2], [y1, y2])
    # 绘制支持向量
    for i, alpha in enumerate(chenzi_mat):
        if abs(alpha) > 1e-3:
            x, y = dataset[i]
            ax.scatter([x], [y], s=180, c='none', alpha=0.7, linewidth=2, edgecolor='black')
    plt.show()
