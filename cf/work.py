# coding=utf-8
import os
import numpy as np


def load_data(file_path):
    '''导入用户商品数据
    input:  file_path(string):用户商品数据存放的文件
    output: data(mat):用户商品矩阵
    '''
    f = open(file_path)
    data = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        tmp = []
        for x in lines:
            if x != "-":
                tmp.append(float(x))  # 直接存储用户对商品的打分
            else:
                tmp.append(2)
        data.append(tmp)
    f.close()

    return np.mat(data)


if __name__ == "__main__":
    rMat = load_data("R.txt")
    qMat = load_data("Q.txt")
    studentNum, n = np.shape(rMat)
    aList = []
    for i in xrange(studentNum):
        student = rMat[i]
        _m, itemsNum = np.shape(student)
        sMat = np.zeros(shape=(1, itemsNum))
        for j in xrange(itemsNum):
            item = student[:, j]
            if item == 1:
                sMat += qMat[j]
        for k in xrange(itemsNum):
            s = sMat[:, k]
            if s >= 2:
                sMat[:, k] = 1
        aList.append(sMat)
    print(aList)
