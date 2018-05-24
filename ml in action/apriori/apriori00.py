#!/usr/bin/python
# coding: utf8
from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def singleGoodsSet(goodList):
    singleGoodsSet = []
    for goods in goodList:
        for good in goods:
            if not [good] in singleGoodsSet:
                singleGoodsSet.append([good])
    singleGoodsSet.sort()
    return map(frozenset, singleGoodsSet)


def getMyGoodsByCondition(goodsList, singleGoodsSet, condition):
    goodsByCounts = {}  # ssCnt 临时存放选数据集 Ck 的频率. 例如: a->10, b->5, c->8
    for goods in goodsList:
        for singleGoods in singleGoodsSet:
            if singleGoods.issubset(goods):  # s.issubset(t)  测试是否 s 中的每一个元素都在 t 中
                if not goodsByCounts.has_key(singleGoods):
                    goodsByCounts[singleGoods] = 1
                else:
                    goodsByCounts[singleGoods] += 1
    goodsList_size = float(len(goodsList))
    myGoodsList = []  # 存放满足我最小支持度的goods
    myGoodsMap = {}  # 存放所有遍历的goods和其对应的支持度
    for goodsName in goodsByCounts:
        ifMyGoods = goodsByCounts[goodsName] / goodsList_size  # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        if ifMyGoods >= condition:
            myGoodsList.insert(0, goodsName)  # 这些支持度的是我需要的
        myGoodsMap[goodsName] = ifMyGoods
    return myGoodsList, myGoodsMap


# 输入频繁项集列表 Lk 与返回的元素个数 k，然后输出所有可能的候选项集 Ck
def goodsGroupByK(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[: k - 2]
            L2 = list(Lk[j])[: k - 2]
            L1.sort()
            L2.sort()
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


# 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
def apriori(goodsList, condition=0.5):
    oneGoodsSet = singleGoodsSet(goodsList)  # 不重复的单个集合
    goodsSet = map(set, goodsList)
    myGoodsList, myGoodsMap = getMyGoodsByCondition(goodsSet, oneGoodsSet, condition)
    allMyGoodsList = [myGoodsList]
    k = 2
    while (len(allMyGoodsList[k - 2]) > 0):
        # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}.
        # 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        goodsGroupBy_k = goodsGroupByK(allMyGoodsList[k - 2], k)
        k_myGoodsList, k_myGoodsMap = getMyGoodsByCondition(goodsSet, goodsGroupBy_k, condition)
        myGoodsMap.update(k_myGoodsMap)  # 如果字典没有，就追加元素，如果有，就更新元素
        if len(k_myGoodsList) == 0:
            break
        allMyGoodsList.append(k_myGoodsList)
        k += 1
    return allMyGoodsList, myGoodsMap


def generateRules(allMyGoodsList, myGoodsMap, condition=0.5):
    confidenceList = []  # 计算出来的可信度或者置信度的列表
    allMyGoodsListSize = len(allMyGoodsList)
    for i in range(1, allMyGoodsListSize):  # 从下标1开始是因为下标为0都是一个goods，这里至少要两个才能计算
        myGoodsList = allMyGoodsList[i]
        for goodsGroup in myGoodsList:
            goodsGroupSet = [frozenset([item]) for item in goodsGroup]
            if (i > 1):
                rulesFromConseq(goodsGroup, goodsGroupSet, myGoodsMap, confidenceList, condition)
            else:
                calcConf(goodsGroup, goodsGroupSet, myGoodsMap, confidenceList, condition)
    return confidenceList


# 计算可信度
def calcConf(goodsGroup, goodsGroupSet, myGoodsMap, confidenceList, condition=0.7):
    # 记录可信度大于最小可信度（minConf）的集合
    confidenceInfo = []
    for oneGoods in goodsGroupSet:
        a2b = myGoodsMap[goodsGroup]  # goodsGroup里有两个商品a,b.它俩在一起的支持度为a2b
        a = myGoodsMap[goodsGroup - oneGoods]  # b的支持度
        # 由购买a->b倒推购买b就会购买a（b->a）的可能性为b2a,这里是指可信度或者置信度(confidence)
        b2a = a2b / a
        if b2a >= condition:
            print goodsGroup - oneGoods, '-->', oneGoods, 'confidence:', b2a
            confidenceList.append((goodsGroup - oneGoods, oneGoods, b2a))
            confidenceInfo.append(oneGoods)
    return confidenceInfo


def rulesFromConseq(goodsGroup, goodsGroupSet, myGoodsMap, confidenceList, condition=0.7):
    m = len(goodsGroupSet[0])
    if (len(goodsGroup) > (m + 1)):
        goodsGroup_k = goodsGroupByK(goodsGroupSet, m + 1)
        goodsGroup_k = calcConf(goodsGroup, goodsGroup_k, myGoodsMap, confidenceList, condition)
        if (len(goodsGroup_k) > 1):
            rulesFromConseq(goodsGroup, goodsGroup_k, myGoodsMap, confidenceList, condition)


def aprioriGen(Lk, k):  # creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2];
            L2 = list(Lk[j])[:k - 2]
            L1.sort();
            L2.sort()
            if L1 == L2:  # if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j])  # set union
    return retList


if __name__ == "__main__":
    dataSet = loadDataSet()
    allMyGoodsList, myGoodsMap = apriori(dataSet)
    rules = generateRules(allMyGoodsList, myGoodsMap)
