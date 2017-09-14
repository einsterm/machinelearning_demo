# --*-- encoding:utf-8 --*--

from numpy import *


def loadDataSet():
    dataSet = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 0
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],  # 1
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],  # 0
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],  # 1
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],  # 0
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']  # 1
    ]
    labels = [0, 1, 0, 1, 0, 1]  # 1 is abusive,0 not
    return dataSet, labels


def distinctDataList(dataSet):
    vocabSet = set([])  # 对数据去重
    for docment in dataSet:
        vocabSet = vocabSet | set(docment)  # 两个集合的并集
    return list(vocabSet)  # 生成词袋


# 生成词袋矩阵，统计testWord在wordPackage里面出现的次数
def createWordM(wordPackage, testWords):
    wordM = [0] * len(wordPackage)  # 创建一个和词袋长度一样的0矩阵
    for word in testWords:
        if word in wordPackage:
            wordIdx = wordPackage.index(word)  # 返回该词在词袋里的下标
            wordM[wordIdx] = 1
    return wordM


# trainM 是整个文档个每个单词的分布
def trainNB(trainM, labels):
    trainLen = len(labels)  # 矩阵的长度
    allWord_num = len(trainM[0])  # 整个文档词汇不重复的个数
    p0_num = ones(allWord_num)
    p1_num = ones(allWord_num)  # 整个文档中，出现在类别1中的所有词出现的次数，默认所有词出现1次，如果设为0次，在作乘积时影响结果
    p0_sum = 2.0
    p1_sum = 2.0  # 整个文档中，出现在类别1中的所有词的个数
    for i in range(trainLen):
        listM = trainM[i]  # 第i行，在整个词袋里的分布
        if labels[i] == 1:
            p1_num += listM  # 两个矩阵相加，得到是矩阵中每个词出现的次数
            p1_sum += sum(listM)  # 第i行，在类别1中，词的个数
        else:
            p0_num += listM
            p0_sum = sum(listM)
    # 每个词在类别1中出现的次数/类别1所有词的数量,之所以要加个log（求对数）,是因为两个数求商可能得到不精确的四舍五入值0
    # 因为如果 a>b 则 log(a)>log(b) 如果需要求a*b,则求log(a)+log(b)，因为log(a*b)=log(a)+log(b)
    p1 = log(p1_num / p1_sum)
    p0 = log(p0_num / p0_sum)
    pAbusive = sum(labels) / float(trainLen)  # 出现1类词的概率
    return p0, p1, pAbusive  # p0,p1即所有词在类别0,1出现概率的对数值


# P(Ci|Wi)=P(Wi|Ci)*P(Ci)/P(Wi)
# P(Ci)=类别0,1在所有类别数目的概率,比如所有类别有10个，其中类别1占的数量是5,则概率是0.5
#P(Wi)=各个词出现的次数/所有词的数量。这个值对于不同类别，它的值是固定的，所以求P(Ci|Wi)，只需要求P(Wi|Ci)*P(Ci)
def classfiyNB(testDataM, p0, p1, p_1):
    p1 = sum(testDataM * p1) + log(p_1)
    p0 = sum(testDataM * p0) + log(1.0 - p_1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    dataSet, labels = loadDataSet()
    distinctList = distinctDataList(dataSet)
    trainMat = []
    for doc in dataSet:
        docinlist = createWordM(distinctList, doc)
        trainMat.append(docinlist)
    p0, p1, pAbusive = trainNB(array(trainMat), array(labels))
    testData = ['love', 'my', 'dog']
    thisDoc = array(createWordM(distinctList, testData))
    print classfiyNB(thisDoc, p0, p1, pAbusive)
    testData = ['garbage', 'stupid']
    thisDoc = array(createWordM(distinctList, testData))
    print classfiyNB(thisDoc, p0, p1, pAbusive)


def getTrainM(dataSet):
    distinctList = distinctDataList(dataSet)  # 合并矩阵，并且去重
    trainM = []
    for arr in dataSet:
        list = createWordM(distinctList, arr)  # 统计矩阵中每个词出现的位置
        trainM.append(list)
    return trainM


if __name__ == '__main__':
    # dataSet, labels = loadDataSet()
    # trainMat = getTrainM(dataSet)
    # p0Vect, p1Vect, pAbusive = trainNB0(trainMat, labels)
    # print p0Vect
    # print p1Vect
    # print pAbusive
    # testingNB()
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8, 9]]
    print array(a) * array(b)
