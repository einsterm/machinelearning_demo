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
    goodsByCounts = {}  # {行：出现次数}
    for goodsList in allGoodsList:  # 第一次遍历
        for goods in goodsList:
            goodsByCounts[goods] = goodsByCounts.get(goods, 0) + allGoodsList[goodsList]
    for k in goodsByCounts.keys():
        if goodsByCounts[k] < condition:
            del (goodsByCounts[k])

    mySingleGoodsSet = set(goodsByCounts.keys())  # 满足出现频率的goods
    if len(mySingleGoodsSet) == 0:
        return None, None
    for k in goodsByCounts:
        goodsByCounts[k] = [goodsByCounts[k], None]  # 格式化:{元素key: [元素次数count, None]},这里的None为childrenNode

    rootTree = treeNode('Null Set', 1, None)
    for goodsGroupByRow, count in allGoodsList.items():  # 第二次遍历，对每一行遍历
        _goodsByCount = {}
        for goods in goodsGroupByRow:  # goodsGroupByRow代表一行
            if goods in mySingleGoodsSet:  # 该goods是否在我的条件范围内（频率）
                _goodsByCount[goods] = goodsByCounts[goods][0]
        if len(_goodsByCount) > 0:
            orderByCountGoods = [v[0] for v in sorted(_goodsByCount.items(), key=lambda p: p[1], reverse=True)]  # 对goods的数量由大到小排序，数量相等的按字母顺序
            updateNode(orderByCountGoods, rootTree, goodsByCounts, count)
    return rootTree, goodsByCounts


def updateNode(orderByCountGoods, parentTree, goodsByCount, count):
    maxCountGoodsName = orderByCountGoods[0]  # 出现频率最高的那个goods
    if maxCountGoodsName in parentTree.children:
        parentTree.children[maxCountGoodsName].inc(count)
    else:
        parentTree.children[maxCountGoodsName] = treeNode(maxCountGoodsName, count, parentTree)
        nextParentNode = goodsByCount[maxCountGoodsName][1]  # 这里刚才开始默认为None
        thisNode = parentTree.children[maxCountGoodsName]
        if nextParentNode == None:
            goodsByCount[maxCountGoodsName][1] = thisNode
        else:
            updateNodeLink(nextParentNode, thisNode)
    if len(orderByCountGoods) > 1:
        thisNode = parentTree.children[maxCountGoodsName]
        updateNode(orderByCountGoods[1::], thisNode, goodsByCount, count)


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


def ascendTree(leafNode, prefixPath):
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(treeNode):
    condPats = {}
    # 对 treeNode的link进行循环
    while treeNode is not None:
        prefixPath = []
        # 寻找改节点的父节点，相当于找到了该节点的频繁项集
        ascendTree(treeNode, prefixPath)
        # 避免 单独`Z`一个元素，添加了空节点
        if len(prefixPath) > 1:
            # 对非basePat的倒叙值作为key,赋值为count数
            # prefixPath[1:] 变frozenset后，字母就变无序了
            # condPats[frozenset(prefixPath)] = treeNode.count
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # 递归，寻找改节点的下一个 相同值的链接节点
        treeNode = treeNode.nodeLink
        # print treeNode
    return condPats


def mineTree(goodsByCounts, condition, preFix, freqItemList):
    singleGoodsSet = [v[0] for v in sorted(goodsByCounts.items(), key=lambda p: p[1])]
    for goods in singleGoodsSet:
        newFreqSet = preFix.copy()
        newFreqSet.add(goods)
        freqItemList.append(newFreqSet)
        findPath = goodsByCounts[goods][1]
        allGoodsList = findPrefixPath(findPath)
        myTree, myGoodsByCounts = createTree(allGoodsList, condition)
        if myGoodsByCounts != None:
            print 'conditional tree for: ', newFreqSet
            myTree.disp(1)
            mineTree(myGoodsByCounts, condition, newFreqSet, freqItemList)


if __name__ == '__main__':
    dataSet = loadSimpDat()
    initSet = createInitSet(dataSet)
    tree, goodsByCounts = createTree(initSet, 3)
    # tree.disp()
    condPats = findPrefixPath(goodsByCounts['x'][1])
    # print(condPats)
    freqItemList = []
    mineTree(goodsByCounts, 3, set([]), freqItemList)
