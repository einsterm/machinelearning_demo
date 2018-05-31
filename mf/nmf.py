# coding:UTF-8

import numpy as np
from mf import load_data, save_file, prediction, top_k

'''
非负矩阵分解算法
'''


def train(V, k, maxCycles, e):
    m, n = np.shape(V)
    # 1、初始化矩阵
    W = np.mat(np.random.random((m, k)))
    H = np.mat(np.random.random((k, n)))

    # 2、非负矩阵分解
    for step in xrange(maxCycles):
        V_pre = W * H
        E = V - V_pre
        err = 0.0
        for i in xrange(m):
            for j in xrange(n):
                err += E[i, j] * E[i, j]

        if err < e:
            break
        if step % 1000 == 0:
            print "\titer: ", step, " loss: ", err

        a = W.T * V
        b = W.T * W * H
        for i in xrange(k):
            for j in xrange(n):
                if b[i, j] != 0:
                    H[i, j] = H[i, j] * a[i, j] / b[i, j]

        c = V * H.T
        d = W * H * H.T
        for i in xrange(m):
            for j in xrange(k):
                if d[i, j] != 0:
                    W[i, j] = W[i, j] * c[i, j] / d[i, j]

    return W, H


if __name__ == "__main__":
    # 1、导入用户商品矩阵
    print "----------- 1、load data -----------"
    V = load_data("data.txt")
    # 2、非负矩阵分解
    print "----------- 2、training -----------"
    W, H = train(V, 5, 10000, 1e-5)
    # 3、保存分解后的结果
    print "----------- 3、save decompose -----------"
    save_file("W", W)
    save_file("H", H)
    # 4、预测
    print "----------- 4、prediction -----------"
    predict = prediction(V, W, H, 0)
    # 进行Top-K推荐
    print "----------- 5、top_k recommendation ------------"
    top_recom = top_k(predict, 2)
    print top_recom
    print W * H
