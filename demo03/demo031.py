# --*-- encoding:utf-8 --*--

from  numpy import *
import kNN
import matplotlib.pyplot as plt


def readDataFile():
    fr = open("data1")
    arrayLines = fr.readlines()
    numberOflines = len(arrayLines)
    returnMat = zeros((numberOflines, 3))
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
    ax.scatter(dataMat[:, 0], dataMat[:, 1], 15.0 * array(labels), 15.0 * array(labels))
    plt.show()


def autoNorm(dataSet):
    minValsMat = dataSet.min(0)  # 0 作用是按矩阵按列取
    maxValsMat = dataSet.max(0)
    distanceMat = maxValsMat - minValsMat  # 矩阵各列的差
    normDataSet = zeros(shape(dataSet))  # 复制一个结构相同的0矩阵
    rowsSize = dataSet.shape[0]  # 矩阵列的长度
    minValsMatCopy = tile(minValsMat, (rowsSize, 1))  # 产生一个最小值的矩阵
    normDataSet = dataSet - minValsMatCopy  # 两个矩阵求差
    distanceValsMat = tile(distanceMat, (rowsSize, 1))  # 产生一个差值矩阵
    normDataSet = normDataSet / distanceValsMat
    return normDataSet, distanceMat, minValsMat


if __name__ == "__main__":
    returnMat, classLabelVector = readDataFile()
    # show(returnMat, classLabelVector)
    normDataSet, distanceMat, minValsMat = autoNorm(returnMat)
    print normDataSet

