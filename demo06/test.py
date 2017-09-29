# --*-- encoding:utf-8 --*--

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()  # 矩阵转置，行变成列，列变成行
    rows, cloumns = shape(dataMatrix)  # 取矩阵的长度
    alpha = 0.001
    maxCycles = 500
    weightsMat = ones((cloumns, 1))
    for k in range(maxCycles):
        cloumnAddVal = dataMatrix * weightsMat  # 把每一行的属性相加起来
        hMat = sigmoid(cloumnAddVal)
        errorMat = (labelMat - hMat)
        dataMatrixT = dataMatrix.transpose()  # 把每一列都抽出来
        dataMatrix1 = alpha * dataMatrixT
        dataMatrix2 = dataMatrix1 * errorMat
        weightsMat = weightsMat + dataMatrix2
    return weightsMat


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights.getA())
    # a = mat([[1, 2, 3], [1, 1, 1]])
    # b = mat([[4], [5], [6]])
    # print a * b
