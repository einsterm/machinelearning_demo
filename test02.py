# -*- encoding:utf-8 -*-
import math


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['surfacing', 'flippers']
    return dataSet, labels


def calcShang(dataSet):
    dataSetLen = len(dataSet)
    resultMap = {}
    for list in dataSet:
        result = list[-1];
        if result not in resultMap.keys():
            resultMap[result] = 0
        resultMap[result] += 1
        shang = 0.0
        for key in resultMap:
            prop = float(resultMap[key]) / dataSetLen
            shang -= prop * math.log(prop, 2)
    return shang


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for list in dataSet:
        if list[axis] == value:
            reduceVec = list[:axis]
            reduceVec.extend(list[axis + 1:])
            retDataSet.append(reduceVec)
    return retDataSet

#计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for vec in dataSet:
        currentLabel = vec[-1]
        if currentLabel not in labelCounts.keys():  #为所有可能的分类建立字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob,2)
    return shannonEnt





#按照给定的特征划分数据集
def splitDataset(dataSet,axis,value):
    retDataset = []     #符合特征的数据
    for vec in dataSet:
        if vec[axis] == value:       #数据特征符合要求
            reducedVec = vec[:axis]  #提取该数据的剩余特征
            reducedVec.extend(vec[axis+1:])   #将两列表合成一个列表
            retDataset.append(reducedVec)
    return retDataset

#选择最好的数据集划分方式
def chooseBestFeatureToSplit1(dataSet):
    numFeatures = len(dataSet[0]) - 1     #特征数
    baseEntropy = calcShannonEnt(dataSet) #计算原始熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in xrange(numFeatures):
        featList = [eg[i] for eg in dataSet] #分类标签列表
        uniqueVals = set(featList)           #构建集合去重
        newEntropy = 0.0
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataset(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet)) #出现比例
            newEntropy += prob * calcShannonEnt(subDataSet)
        #计算最好的信息增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    shang = calcShang(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in xrange(numFeatures):
        featList = [eg[i] for eg in dataSet]
        uniqueVals = set(featList)
        newShang = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newShang += prob * calcShang(subDataSet)
        infoGain = shang - newShang
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


if __name__ == '__main__':
    dataSet, label = createDataSet()
    shang = chooseBestFeatureToSplit1(dataSet)
    print shang
