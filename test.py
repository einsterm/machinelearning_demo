# --*-- encoding:utf-8 --*--

import math
import operator
import matplotlib.pyplot as plt


# 计算熵
def calcShannonEnt(dataSet):
    dataLen = len(dataSet)
    labels = {}
    for featureVector in dataSet:
        currentLabel = featureVector[-1]
        if currentLabel not in labels.keys():
            labels[currentLabel] = 0
        labels[currentLabel] += 1
    shannon = 0.0
    for key in labels:
        prob = float(labels[key]) / dataLen
        shannon += -prob * math.log(prob, 2)
    return shannon


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 按矩阵的列切分，idx是矩阵的列下标，value是匹配到该下标的值
def splitDataSet(dataSet, idx, value):
    retDataSet = []  # 切分剩下的矩阵
    for featureList in dataSet:
        val = featureList[idx]
        if val == value:
            rfv = featureList[:idx]  # 下标值 看“:”取那边的值
            rfvVal = featureList[idx + 1:]  # 右边的值
            rfv.extend(rfvVal)
            retDataSet.append(rfv)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 得到所有属性，特征
    baseEntropy = calcShannonEnt(dataSet)  # 最大的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for rowIdx in range(numFeatures):
        featureValueList = [example[rowIdx] for example in dataSet]
        featureValueSet = set(featureValueList)
        newEntropy = 0.0
        for featureValue in featureValueSet:
            subDataSet = splitDataSet(dataSet, rowIdx, featureValue)
            prob = len(subDataSet) / float(len(dataSet))
            subShannon = calcShannonEnt(subDataSet)  # 剩下所有数据的熵
            newEntropy += prob * subShannon  # 该分类在总的数据下熵，值越小，说明信息的无序程度越小
        infoGain = baseEntropy - newEntropy  # 值越大越好，说明newEntropy越小，baseEntropy不变
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = rowIdx
    return bestFeature


# 挑出labels里面，出现次数最多的label
def getMaxcountLabel(labels):
    labelMap = {}  # key=标签，value=出现的次数
    for vote in labels:
        if vote not in labelMap.keys():
            labelMap[vote] = 0
        labelMap[vote] += 1
    iteritems = labelMap.iteritems()
    itemgetter = operator.itemgetter(1)
    sortedLabelMap = sorted(iteritems, key=itemgetter, reverse=True)
    return sortedLabelMap[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 取出所有标签

    label = classList[0]  # 取出第一个标签
    labelCount = classList.count(label)  # 取出label标签的个数

    if labelCount == len(classList):  # 如果只剩下同一类标签了，就不能再划分类别了，直接返回
        return classList[0]
    if len(dataSet[0]) == 1:  # 只有一个标签了
        return getMaxcountLabel(classList)
    bestFeatureRowIdx = chooseBestFeatureToSplit(dataSet)  # 获取最佳分类列的下标
    bestFeatureLabel = labels[bestFeatureRowIdx]  # 对应的标签
    myTree = {bestFeatureLabel: {}}
    del (labels[bestFeatureRowIdx])  # 剔除已经知道
    allFeatureValues = [example[bestFeatureRowIdx] for example in dataSet]  # 通过列的下标，获取该下标对应列的所有值

    bestFeatureRowVals = set(allFeatureValues)
    for rowValue in bestFeatureRowVals:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeatureRowIdx, rowValue)
        myTree[bestFeatureLabel][rowValue] = createTree(subDataSet, subLabels)
    return myTree


descNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode(U'node1', (0.5, 0.1), (0.1, 0.5), descNode)
    plotNode(U'node2', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    # dataSet[0][-1] = 'maybe'
    # shannon = calcShannonEnt(dataSet)
    # print shannon
    # res = splitDataSet(dataSet, 1, 1)
    # print res
    # best = chooseBestFeatureToSplit(dataSet)
    # print best

    # print majorityCnt(a)
    createPlot()
