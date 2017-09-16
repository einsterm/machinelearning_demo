# --*-- encoding:utf-8 --*--

import feedparser
import operator
import emaildemo as email
import bayes
from numpy import *
import math


def getTopWordMaps(distinctList, allWord):
    wordCountMap = {}
    for word in distinctList:
        wordCountMap[word] = allWord.count(word)  # 统计词出现的次数,key=词，value=该词出现的次数
    sortedWordMap = sorted(wordCountMap.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedWordMap[:30]  # 取出前30个


def localWords(feed1, feed0):
    docList = []
    labels = []
    fullText = []
    feed0Len = len(feed0['entries'])
    feed1Len = len(feed1['entries'])

    minLen = min(feed0Len, feed1Len)
    for i in range(minLen):
        str1 = feed1['entries'][i]['summary']
        wordList = email.textParse(str1)
        docList.append(wordList)
        fullText.extend(wordList)
        labels.append(1)

        str0 = feed0['entries'][i]['summary']
        wordList = email.textParse(str0)
        docList.append(wordList)
        fullText.extend(wordList)
        labels.append(0)

    distinctList = bayes.distinctDataList(docList)
    top30WordMaps = getTopWordMaps(distinctList, fullText)
    for wordMap in top30WordMaps:
        hotWord = wordMap[0]  # 取每个map的key
        if hotWord in distinctList:
            distinctList.remove(hotWord)  # 去除高频词，比如，a,the,this等无意义的词
    trainingSet = range(2 * minLen)
    testSet = []
    for i in range(20):
        randomIndex = int(random.uniform(0, len(trainingSet)))  # 随机获取几个词用于测试
        testSet.append(trainingSet[randomIndex])
        del (trainingSet[randomIndex])
    trainMat = []
    trainLabels = []
    for docIndex in trainingSet:
        trainMat.append(bayes.createWordM(distinctList, docList[docIndex]))
        trainLabels.append(labels[docIndex])
    p0V, p1V, pSpam = bayes.trainNB(array(trainMat), array(trainLabels))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.createWordM(distinctList, docList[docIndex])
        if bayes.classfiyNB(array(wordVector), p0V, p1V, pSpam) != labels[docIndex]:
            errorCount += 1
    print float(errorCount) / len(testSet)
    return distinctList, p0V, p1V


def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p0V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    for word in sortedSF:
        print word[0]
    print "--------------------------------------------------"
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    for word in sortedNY:
        print word[0]


if __name__ == '__main__':
    f1 = feedparser.parse('http://newyork.craigslist.org/search/stp?format=rss')
    f2 = feedparser.parse('http://sfbay.craigslist.org/search/stp?format=rss')
    getTopWords(f1, f2)
