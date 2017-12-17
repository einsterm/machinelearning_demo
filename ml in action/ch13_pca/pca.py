#!/usr/bin/python
# coding: utf-8
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)  # 求列的平均值
    meanRemoved = dataMat - meanVals  # 每一列去减去平均值
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算特征值
    eigValInd = argsort(eigVals)  # argsort返回了排序索引
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 取得特征值最大的那个索引
    redEigVects = eigVects[:, eigValInd]  # 取得特征值最大的那个特征向量
    lowDDataMat = meanRemoved * redEigVects  # 数据降维
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()


if __name__ == '__main__':
    dataMat = loadDataSet("testSet10.txt")
    # print(shape(dataMat))
    lowDDataMat, reconMat = pca(dataMat, 1)
    print(shape(lowDDataMat))
    print(shape(reconMat))
    show_picture(dataMat, reconMat)


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # set NaN values to mean
    return datMat
