# --*-- encoding:utf-8 --*--

from  numpy import *
import operator
import matplotlib.pyplot as plt

'''KNN算法的基本实现，给出一个点poinX 求该点与训练集中距离最近的点'''


# 创建训练数据
def createDataSet():
    group = array([[1.0, 2.3], [1.0, 3.1], [4.4, 6.1], [5.2, 7.5]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 展示数据分布
def show(dataSet, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    index = 0
    for point in dataSet:
        if labels[index] == 'A':
            ax.scatter(point[0], point[1], c='blue', marker='o', linewidths=0, s=300)
            plt.annotate("(" + str(point[0]) + "," + str(point[1]) + ")", xy=(point[0], point[1]))
        else:
            ax.scatter(point[0], point[1], c='red', marker='o', linewidths=0, s=300)
            plt.annotate("(" + str(point[0]) + "," + str(point[1]) + ")", xy=(point[0], point[1]))
        index += 1
    plt.show()


# 计算KNN距离
def calcKnn(pointX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 求矩阵的“行数”
    diffMat = tile(pointX, (dataSetSize, 1)) - dataSet  # 把pointX与训练的矩阵相减，得到差
    sqDiffMat = diffMat ** 2  # 求差的平方
    distanceArray = sqDiffMat.sum(axis=1)  # 矩阵相量求和
    distanceArrayRes = distanceArray ** 0.5  # 开平方
    sortedIdxArray = distanceArrayRes.argsort()  # argsort函数返回的是数组值从小到大的 索引值
    labelsMap = {}  # 存放的是 key=类别，value=数量
    for i in range(k):
        sortLabel = labels[sortedIdxArray[i]]  # 排序后的label
        labelsMap[sortLabel] = labelsMap.get(sortLabel, 0) + 1  # 统计类别总数量
    sortedLabelMap = sorted(labelsMap.iteritems(), key=operator.itemgetter(1), reverse=True)  # 再次排序
    return sortedLabelMap[0][0]  # 取第一个元素的key


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    point = [2, 3]
    # show(dataSet, labels)
    res = calcKnn(point, dataSet, labels, 3)
    print res
