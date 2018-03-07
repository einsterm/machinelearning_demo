#!/usr/bin/python
# coding:utf8


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        # needs to be updated
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print '-' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(allGoodsList, condition=1):
    goodsByCountsMap = {}  # {行：出现次数}
    for goodsList in allGoodsList:  # 第一次遍历
        for goods in goodsList:
            goodsByCountsMap[goods] = goodsByCountsMap.get(goods, 0) + allGoodsList[goodsList]
    for k in goodsByCountsMap.keys():
        if goodsByCountsMap[k] < condition:
            del (goodsByCountsMap[k])

    mySingleGoodsSet = set(goodsByCountsMap.keys())  # 满足出现频率的goods
    if len(mySingleGoodsSet) == 0:
        return None, None
    for k in goodsByCountsMap:
        # 格式化:{元素: [元素次数count, childrenNode]}
        goodsByCountsMap[k] = [goodsByCountsMap[k], None]

    rootTree = treeNode('Null Set', 1, None)
    for goodsGroupByRow, count in allGoodsList.items():  # 第二次遍历，对每一行遍历
        _goodsByCount = {}  # 格式{'y':3,'x':2}
        for goods in goodsGroupByRow:  # goodsGroupByRow代表一行
            if goods in mySingleGoodsSet:  # 该goods是否在我的条件范围内（频率）
                _goodsByCount[goods] = goodsByCountsMap[goods][0]
        if len(_goodsByCount) > 0:
            # 对goods的数量由大到小排序，数量相等的按字母顺序
            _orderByCountGoods = [v[0] for v in sorted(_goodsByCount.items(), key=lambda p: p[1], reverse=True)]
            updateNode(_orderByCountGoods, rootTree, goodsByCountsMap, count)
    return rootTree, goodsByCountsMap


def updateNode(_orderByCountGoodsList, parentTree, goodsByCountMap, count):
    maxCountGoodsName = _orderByCountGoodsList[0]  # 出现频率最高的那个goods
    if maxCountGoodsName in parentTree.children:
        parentTree.children[maxCountGoodsName].inc(count)
    else:
        parentTree.children[maxCountGoodsName] = treeNode(maxCountGoodsName, count, parentTree)
        nextParentNode = goodsByCountMap[maxCountGoodsName][1]  # 这里刚才开始默认为None
        thisNode = parentTree.children[maxCountGoodsName]
        if nextParentNode == None:
            goodsByCountMap[maxCountGoodsName][1] = thisNode
        else:
            updateNodeLink(nextParentNode, thisNode)
    if len(_orderByCountGoodsList) > 1:
        thisNode = parentTree.children[maxCountGoodsName]
        _orderByCountGoodsList_ = _orderByCountGoodsList[1::]
        updateNode(_orderByCountGoodsList_, thisNode, goodsByCountMap, count)


def updateNodeLink(nodeToTest, targetNode):
    while (nodeToTest.nodeLink is not None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        if not retDict.has_key(frozenset(trans)):
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1
    return retDict


def ascendTree(node, parentPath):
    if node.parent is not None:
        parentPath.append(node.name)
        ascendTree(node.parent, parentPath)


def findParentPath(childrenNode):
    condPats = {}
    while childrenNode is not None:
        parentPath = []
        ascendTree(childrenNode, parentPath)
        if len(parentPath) > 1:
            condPats[frozenset(parentPath[1:])] = childrenNode.count
        childrenNode = childrenNode.nodeLink
    return condPats


def mineTree(goodsByCountsMap, condition, preFix, freqItemList):
    singleGoodsSet = [v[0] for v in sorted(goodsByCountsMap.items(), key=lambda p: p[1])]
    for goods in singleGoodsSet:
        newFreqSet = preFix.copy()
        newFreqSet.add(goods)
        freqItemList.append(newFreqSet)
        findPath = goodsByCountsMap[goods][1]
        allGoodsList = findParentPath(findPath)
        myTree, myGoodsByCounts = createTree(allGoodsList, condition)
        if myGoodsByCounts != None:
            print 'conditional tree for: ', newFreqSet
            myTree.disp(1)
            mineTree(myGoodsByCounts, condition, newFreqSet, freqItemList)


if __name__ == '__main__':
    dataSet = loadSimpDat()
    initSet = createInitSet(dataSet)
    rootTree, goodsByCountsMap = createTree(initSet, 3)
    # tree.disp()
    childrenNode = goodsByCountsMap['r'][1]
    parentPath = findParentPath(childrenNode)
    print(parentPath)
    freqItemList = []
    mineTree(goodsByCountsMap, 3, set([]), freqItemList)
