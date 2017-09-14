# --*-- encoding:utf-8 --*--
import bayes
import re
import random
from numpy import *


def textParse(bigString):
    list = re.split(r'\W*', bigString)
    return [w.lower() for w in list if len(w) > 2]


def spamTest():
    docList = []
    labels = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        labels.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        labels.append(0)

    trainingIdxSet = range(50)
    testIdxSet = []

    for i in range(10):
        randomIndex = int(random.uniform(0, len(trainingIdxSet)))
        testIdxSet.append(trainingIdxSet[randomIndex])
        del (trainingIdxSet[randomIndex])

    trainingM = []
    trainingLabels = []
    distinctList = bayes.distinctDataList(docList)

    for docIndex in trainingIdxSet:
        wordM = bayes.createWordM(distinctList, docList[docIndex])
        trainingM.append(wordM)
        trainingLabels.append(labels[docIndex])
    p0V, p1V, pSpam = bayes.trainNB(array(trainingM), array(trainingLabels))

    errorCount = 0
    for docIndex in testIdxSet:
        testDataM = bayes.createWordM(distinctList, docList[docIndex])
        if bayes.classfiyNB(array(testDataM), p0V, p1V, pSpam) != labels[docIndex]:
            errorCount += 1
    print float(errorCount) / len(testIdxSet)


if __name__ == '__main__':
    spamTest()
