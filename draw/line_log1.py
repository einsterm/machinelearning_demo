# --*-- encoding:utf-8 --*--
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import random
from math import log


def generateData(kMat, meanMat, sigma, dataNum):
    '''
    产生混合高斯模型的数据
    :param kMat: 比例系数
    :param meanMat: 均值
    :param sigma: 标准差
    :param dataNum:数据个数
    :return: 生成的数据
    '''
    dataArray = np.zeros(dataNum, dtype=np.float32)
    gaussNum = len(kMat)  # 高斯分布个数
    for dataIdx in range(dataNum):
        rand = np.random.random()  # 产生[0,1]之间的随机数
        Sum = 0
        index = 0
        while (index < gaussNum):
            Sum += kMat[index]
            if (rand < Sum):
                meanMat_val = meanMat[index]  # 均值
                sigma_val = sigma[index]  # 标准方差
                dataArray[dataIdx] =  np.random.normal(meanMat_val, sigma_val)  # 随机变量它的对应高斯分布值
                break
            else:
                index += 1
    return dataArray


def createData(kMat, meanMat, sigma, dataNum):
    dataMap = {}
    gaussNum = len(kMat)  # 高斯分布个数
    for gaussIdx in range(gaussNum):
        dataLen = int(round(dataNum * float(kMat[gaussIdx])))
        dataArray = np.zeros(dataLen, dtype=np.float32)
        meanMat_val = meanMat[gaussIdx]  # 均值
        sigma_val = sigma[gaussIdx]  # 标准方差
        for i in range(dataLen):
            dataArray[i] = np.random.normal(meanMat_val, sigma_val)
        dataMap[gaussIdx] = dataArray
    return dataMap


def getSumNum(len):
    sum = 0
    idx = 0
    arr = []
    while (idx < len):
        ele = random.randint(10, 20)
        arr.append(ele)
        sum += ele
        idx += 1
    _sum = 0
    _arr = []
    for i in range(len):
        mele = float(float(arr[i]) / float(sum))
        _arr.append(mele)
        _sum += mele

    return _arr


def func1(dataMap, meanMat, sigmaMat):
    xarr = []
    yarr = []
    resMap = {}
    for key in dataMap:
        arr = dataMap[key]
        u = float(meanMat[key])
        sigma = float(sigmaMat[key])

        for i in range(len(arr)):
            x = arr[i]
            y = normfun(x, u, sigma)
            xarr.append(y)
            resMap[y] = log(y, 10)
    _xarr = sorted(xarr)
    for item in _xarr:
        yarr.append(resMap[item])
    return _xarr, yarr


def funca(xArr, thetaMap):
    yArray = []
    for i in range(len(xArr)):
        thetaiArr = thetaMap[i]
        if (i > 0):
            sum = 0
            for j in range(i + 1):
                xi = xArr[j]
                thetai = thetaiArr[j]
                val = xi * thetai
                sum += val
            yArray.append(log(sum, 10))
        else:
            xi = xArr[i]
            thetai = thetaiArr[0]
            val = xi * thetai
            yArray.append(log(val, 10))
    return yArray


def funcb(xArr, thetaMap):
    yArray = []
    for i in range(len(xArr)):
        thetaiArr = thetaMap[i]
        if (i > 0):
            sum = 0
            for j in range(i + 1):
                xi = xArr[j]
                thetai = thetaiArr[j]
                logVal = log(xi, 10)
                val = thetai * logVal
                sum += val
            yArray.append(sum)
        else:
            xi = xArr[i]
            thetai = thetaiArr[0]
            logVal = log(xi, 10)
            val = thetai * logVal
            yArray.append(val)
    return yArray


def normfun(x, mu, sigma):
    return (1. / np.sqrt(2 * np.pi)) * (np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))


def getthetaMap(len):
    thetaMap = {}
    for i in range(len):
        thetaMap[i] = getSumNum(i + 1)
    return thetaMap


if __name__ == '__main__':
    dataLen = 1000
    k = [1]
    mu = [.5]
    sigma = [.02]
    xArr = generateData(k, mu, sigma, dataLen)
    # thetaArr = getSumNum(dataLen)
    xArr = sorted(xArr)

    # thetaArr = sorted(thetaArr)
    thetaMap = getthetaMap(dataLen)

    # thetaMap = {}
    # thetaMap[0] = [1]
    # thetaMap[1] = [0.3, 0.7]
    #
    # xArr = [200, 400]

    yaArray = funca(xArr, thetaMap)
    ybArray = funcb(xArr, thetaMap)
    # xarr, yarr = func1(dataMap, mu, sigma)

    plt.plot(xArr, yaArray, c='b')
    plt.plot(xArr, ybArray, c='r')
    plt.title("XY=-1")
    # xlim(-8, 25)
    # ylim(-8, 25)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)

    plt.show()
