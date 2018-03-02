# --*-- encoding:utf-8 --*--

import numpy as np
import kNN
import matplotlib.pyplot as plt
import scipy


def readDataFile():
    fr = open("data1")
    arrayLines = fr.readlines()
    numberOflines = len(arrayLines)
    returnMat = np.zeros((numberOflines, 3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def show(dataMat, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], 15.0 * np.array(labels), 15.0 * np.array(labels))
    plt.show()


def autoNorm(dataSet):
    minValsMat = dataSet.min(0)  # 0 作用是按矩阵按列取
    maxValsMat = dataSet.max(0)
    distanceMat = maxValsMat - minValsMat  # 矩阵各列的差
    normDataSet = np.zeros(np.shape(dataSet))  # 复制一个结构相同的0矩阵
    rowsSize = dataSet.shape[0]  # 矩阵列的长度
    minValsMatCopy = np.tile(minValsMat, (rowsSize, 1))  # 产生一个最小值的矩阵
    normDataSet = dataSet - minValsMatCopy  # 两个矩阵求差
    distanceValsMat = np.tile(distanceMat, (rowsSize, 1))  # 产生一个差值矩阵
    normDataSet = normDataSet / distanceValsMat
    return normDataSet, distanceMat, minValsMat


def test():
    dataSet, lables = readDataFile()  # 读取数据
    normDataSet, distanceMat, minValsMat = autoNorm(dataSet)  # 归一化特征值
    testRadio = 0.1  # 测试比例
    rowsSize = normDataSet.shape[0]  # 行数
    testMatSize = int(rowsSize * testRadio)  # 测试矩阵比例
    errorCount = 0
    for i in range(testMatSize):
        pointX = normDataSet[i, :]  # 测试的点
        trainSet = scipy.delete(normDataSet, i, 0)  # 去除测试点后的训练数据
        trainLabels = scipy.delete(lables, i, 0)  # 去除测试点后的训练标签
        k = 3
        res = kNN.calcKnn(pointX, trainSet, trainLabels, k)  # 求knn
        realRes = lables[i]
        print "预测是%d,实际是%d" % (res, realRes)
        if res != realRes:
            errorCount += 1
    rate = errorCount / float(testMatSize)  # 计算错误率
    return rate


if __name__ == "__main__":
    # returnMat, classLabelVector = readDataFile()
    # show(returnMat, classLabelVector)
    # normDataSet, distanceMat, minValsMat = autoNorm(returnMat)
    rate = test()
    print rate
