# --*-- encoding:utf-8 --*--

import numpy as np
import kNN
import os


# 把一个文本文件转化为一个一维矩阵
def img2vector(filename):
    returnVect = np.zeros((1, 1024))  # 1 表示行数，转换成的目标矩阵
    fr = open(filename)
    for i in range(32):  # 读取32行 i：行的下标
        lineStr = fr.readline()  # 行
        for j in range(32):  # 读取每一行的前32个字符 i：字符的下标
            val = int(lineStr[j])
            idx = int(32 * i + j)  # 第i行，第j个字母
            returnVect[0, idx] = val  # 0表示行数下标
    return returnVect


def test():
    trainPath = "F:/TDDOWNLOAD/digits/trainingDigits/"
    testPath = "F:/TDDOWNLOAD/digits/testDigits/"
    trianDirArr = os.listdir(trainPath)
    testDirArr = os.listdir(testPath)
    trainSize = len(trianDirArr)
    testSize = len(testDirArr)
    trainMat = np.zeros((trainSize, 1024))
    testMat = np.zeros((testSize, 1024))
    trainLabels = []
    testLabels = []
    for i in range(trainSize):
        fileName = trianDirArr[i]
        label = fileName[0:1]
        trainLabels.append(label)
        mat = img2vector(trainPath + fileName)
        trainMat[i, :] = mat
    for i in range(testSize):
        fileName = testDirArr[i]
        label = fileName[0:1]
        testLabels.append(label)
        mat = img2vector(testPath + fileName)
        testMat[i, :] = mat
    errorCount = 0
    for i in range(testSize):
        pointX = testMat[i, :]
        res = kNN.calcKnn(pointX, trainMat, trainLabels, 3)
        realRes = testLabels[i]
        print "预测是%s,实际是%s" % (res, realRes)
        if res != realRes:
            errorCount += 1
    rate = errorCount / float(testSize)  # 计算错误率
    return rate


if __name__ == '__main__':
    # print img2vector("0_10.txt")
    rate = test()
    print rate
