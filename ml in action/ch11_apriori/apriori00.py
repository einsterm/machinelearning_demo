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
    myGoods = {}
    for goodsName in goodsByCounts:
        ifMyGoods = goodsByCounts[goodsName] / goodsList_size  # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        if ifMyGoods >= condition:
            myGoodsList.insert(0, goodsName)
        myGoods[goodsName] = ifMyGoods
    return myGoodsList, myGoods


# 输入频繁项集列表 Lk 与返回的元素个数 k，然后输出所有可能的候选项集 Ck
def aprioriGen(Lk, k):
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
            # print '-----i=', i, k-2, Lk, Lk[i], list(Lk[i])[: k-2]
            # print '-----j=', j, k-2, Lk, Lk[j], list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            # if first k-2 elements are equal
            if L1 == L2:
                # set union
                # print 'union=', Lk[i] | Lk[j], Lk[i], Lk[j]
                retList.append(Lk[i] | Lk[j])
    return retList


# 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
def apriori(dataSet, minSupport=0.5):
    """apriori（首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。那么满足最小支持度要求的项集构成集合 L1。然后 L1 中的元素相互组合成 C2，C2 再进一步过滤变成 L2，然后以此类推，知道 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度。）

    Args:
        dataSet 原始数据集
        minSupport 支持度的阈值
    Returns:
        L 频繁项集的全集
        supportData 所有元素和支持度的全集
    """
    # C1 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    C1 = singleGoodsSet(dataSet)
    # print 'C1: ', C1
    # 对每一行进行 set 转换，然后存放到集合中
    D = map(set, dataSet)
    # print 'D=', D
    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = getMyGoodsByCondition(D, C1, minSupport)
    # print "L1=", L1, "\n", "outcome: ", supportData

    # L 加了一层 list, L 一共 2 层 list
    L = [L1]
    k = 2
    # 判断 L 的第 k-2 项的数据长度是否 > 0。第一次执行时 L 为 [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]。L[k-2]=L[0]=[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]，最后面 k += 1
    while (len(L[k - 2]) > 0):
        # print 'k=', k, L, L[k-2]
        Ck = aprioriGen(L[k - 2], k)  # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        # print 'Ck', Ck

        Lk, supK = getMyGoodsByCondition(D, Ck, minSupport)  # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
        # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK)
        if len(Lk) == 0:
            break
        # Lk 表示满足频繁子项的集合，L 元素在增加，例如:
        # l=[[set(1), set(2), set(3)]]
        # l=[[set(1), set(2), set(3)], [set(1, 2), set(2, 3)]]
        L.append(Lk)
        k += 1
        # print 'k=', k, len(L[k-2])
    return L, supportData


if __name__ == "__main__":
    dataSet = loadDataSet()
    C1 = singleGoodsSet(dataSet)
    # print(C1)
    D = map(set, dataSet)
    L1, suppData0 = getMyGoodsByCondition(D, C1, 0.5)
    # print(L1)
    L, supportData = apriori(dataSet)
    print(L[0])
    print(L[1])
    print(L[2])