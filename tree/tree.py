# --*-- encoding:utf-8 --*--

import math
import operator
import matplotlib.pyplot as plt
import pickle


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
        p_key = float(labels[key]) / dataLen  # 该key（即该分类）所占有的比重
        l_key = math.log(p_key, 2)  # 根据公式计算a的信息增益
        shannon += -p_key * l_key  # 根据公式计算熵
    return shannon


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 按矩阵的列切分，idx是矩阵的列下标，value是匹配到该下标的值
def splitDataSet(dataSet, idx, value):
    subDataSet = []  # 切分剩下的矩阵
    for oneLine in dataSet:
        val = oneLine[idx]
        if val == value:
            left = oneLine[:idx]  # 一行数据中，按idx下标切分，分为左右两部分数据，这里取是左边数据
            right = oneLine[idx + 1:]  # 右边的值
            left.extend(right)
            subDataSet.append(left)
    return subDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 得到所有属性，特征
    maxEntropy = calcShannonEnt(dataSet)  # 最大的熵
    bestInfoGain = 0.0
    bestFeatureIdx = -1
    for rowIdx in range(numFeatures):
        featureValueList = [example[rowIdx] for example in dataSet]  # 把第rowIdx列所有的值取出来
        featureValueSet = set(featureValueList)  # 该列有几个不同的值
        newEntropy = 0.0
        for featureValue in featureValueSet:
            subDataSet = splitDataSet(dataSet, rowIdx, featureValue)  # 满足特征值为featureValue的字数据集
            p_featureValue = len(subDataSet) / float(len(dataSet))  # 该数据集所占的比重
            subShannon = calcShannonEnt(subDataSet)  # 剩下所有数据的熵
            newEntropy += p_featureValue * subShannon  # 该分类在总的数据下熵，值越小，说明信息的无序程度越小
        infoGain = maxEntropy - newEntropy  # 值越大越好，说明newEntropy越小，baseEntropy不变
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeatureIdx = rowIdx
    return bestFeatureIdx


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


'''下面部分是展示树'''

descNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, descNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


'''读取和存储树'''


def saveTree(inTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inTree, fw)
    fw.close()


def readTree(filename):
    fr = open(filename)
    return pickle.load(fr)


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
    inTree = createTree(dataSet, labels)
    print(inTree)
    createPlot(inTree)
