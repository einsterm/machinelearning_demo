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
    myGoodsList = []
    myGoodsMap = {}
    for goodsName in goodsByCounts:
        ifMyGoods = goodsByCounts[goodsName] / goodsList_size  # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        if ifMyGoods >= condition:
            myGoodsList.insert(0, goodsName)
        myGoodsMap[goodsName] = ifMyGoods
    return myGoodsList, myGoodsMap


# 输入频繁项集列表 Lk 与返回的元素个数 k，然后输出所有可能的候选项集 Ck
def goodsGroupByK(Lk, k):
    """aprioriGen（输入频繁项集列表 Lk 与返回的元素个数 k，然后输出候选项集 Ck。
       例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
       仅需要计算一次，不需要将所有的结果计算出来，然后进行去重操作
       这是一个更高效的算法）

    Args:
        Lk 频繁项集列表
        k 返回的项集元素个数（若元素的前 k-2 相同，就进行合并）
    Returns:
        retList 元素两两合并的数据集
    """

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
    """apriori（首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。那么满足最小支持度要求的项集构成集合 L1。然后 L1 中的元素相互组合成 C2，C2 再进一步过滤变成 L2，然后以此类推，知道 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度。）

    Args:
        dataSet 原始数据集
        minSupport 支持度的阈值
    Returns:
        L 频繁项集的全集
        supportData 所有元素和支持度的全集
    """
    oneGoodsSet = singleGoodsSet(goodsList)
    goodsSet = map(set, goodsList)
    myGoodsList, myGoodsMap = getMyGoodsByCondition(goodsSet, oneGoodsSet, condition)
    allMyGoodsList = [myGoodsList]
    k = 2
    while (len(allMyGoodsList[k - 2]) > 0):
        goodsGroupBy_k = goodsGroupByK(allMyGoodsList[k - 2], k)  # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        k_myGoodsList, k_myGoodsMap = getMyGoodsByCondition(goodsSet, goodsGroupBy_k, condition)
        myGoodsMap.update(k_myGoodsMap)  # 如果字典没有，就追加元素，如果有，就更新元素
        if len(k_myGoodsList) == 0:
            break
        allMyGoodsList.append(k_myGoodsList)
        k += 1
    return allMyGoodsList, myGoodsMap


def generateRules(allMyGoodsList, myGoodsMap, condition=0.7):
    """generateRules

    Args:
        L 频繁项集列表
        supportData 频繁项集支持度的字典
        minConf 最小置信度
    Returns:
        bigRuleList 可信度规则列表（关于 (A->B+置信度) 3个字段的组合）
    """
    confidenceList = []  # 计算出来的可信度或者置信度的列表
    for i in range(1, len(allMyGoodsList)):  # 从下标1开始是因为下标为0都是一个goods，这里至少要两个才能计算
        myGoodsList = allMyGoodsList[i]
        for goodsGroup in myGoodsList:
            goodsGroupSet = [frozenset([item]) for item in goodsGroup]
            if (i > 1):
                rulesFromConseq(goodsGroup, goodsGroupSet, myGoodsMap, confidenceList, condition)
            else:
                calcConf(goodsGroup, goodsGroupSet, myGoodsMap, confidenceList, condition)
    return confidenceList


# 计算可信度   calcConf(goodsGroup, oneGoods, myGoodsMap, bigRuleList, condition)
def calcConf(goodsGroup, goodsGroupSet, myGoodsMap, confidenceList, condition=0.7):
    """calcConf（对两个元素的频繁项，计算可信度，例如： {1,2}/{1} 或者 {1,2}/{2} 看是否满足条件）

    Args:
        freqSet 频繁项集中的元素，例如: frozenset([1, 3])
        H 频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
        supportData 所有元素的支持度的字典
        brl 关联规则列表的空数组
        minConf 最小可信度
    Returns:
        prunedH 记录 可信度大于阈值的集合
    """
    # 记录可信度大于最小可信度（minConf）的集合
    confidenceInfo = []
    for oneGoods in goodsGroupSet:
        a2b = myGoodsMap[goodsGroup]  # goodsGroup里有两个商品a,b.它俩在一起的支持度为a2b
        a = myGoodsMap[goodsGroup - oneGoods]  # b的支持度
        b2a = a2b / a  # 由购买a->b倒推购买b就会购买a（b->a）的可能性为b2a,这里是指可信度或者置信度(confidence)
        if b2a >= condition:
            print goodsGroup - oneGoods, '-->', oneGoods, 'confidence:', b2a
            confidenceList.append((goodsGroup - oneGoods, oneGoods, b2a))
            confidenceInfo.append(oneGoods)
    return confidenceInfo


def rulesFromConseq(goodsGroup, goodsGroupSet, myGoodsMap, confidenceList, condition=0.7):
    """rulesFromConseq

    Args:
        freqSet 频繁项集中的元素，例如: frozenset([2, 3, 5])
        H 频繁项集中的元素的集合，例如: [frozenset([2]), frozenset([3]), frozenset([5])]
        supportData 所有元素的支持度的字典
        brl 关联规则列表的数组
        minConf 最小可信度
    """

    m = len(goodsGroupSet[0])
    if (len(goodsGroup) > (m + 1)):
        goodsGroup_k = goodsGroupByK(goodsGroupSet, m + 1)
        goodsGroup_k = calcConf(goodsGroup, goodsGroup_k, myGoodsMap, confidenceList, condition)
        if (len(goodsGroup_k) > 1):
            rulesFromConseq(goodsGroup, goodsGroup_k, myGoodsMap, confidenceList, condition)


if __name__ == "__main__":
    dataSet = loadDataSet()
    C1 = singleGoodsSet(dataSet)
    # print(C1)
    D = map(set, dataSet)
    L1, suppData0 = getMyGoodsByCondition(D, C1, 0.5)
    # print(L1)
    allMyGoodsList, myGoodsMap = apriori(dataSet)
    # print(L[0])
    # print(L[1])
    # print(L[2])
    # print(supportData)
    rules = generateRules(allMyGoodsList, myGoodsMap)
    print(rules)
