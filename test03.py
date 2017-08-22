# -*- encoding=utf-8 -*-
import csv
import random
import math


def loadCsv(filename):
    lines = csv.reader(open(filename, 'rb'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# 函数假设样本中最后一个属性（-1）为类别值，返回一个类别值到数据样本列表的映射
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# 求平均值
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# 求平均值和标准方差
def stdev(numbers):
    avg = mean(numbers)
    sumVal = sum([pow(x - avg, 2) for x in numbers])  # pow的参数2表示平方
    variance = sumVal / float(len(numbers) - 1)
    return math.sqrt(variance)  # 求平方根


# 按列抽出属性为一个数组，计算该的平均值和标准方差
def summarize(dataset):
    summarize = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summarize[-1]
    return summarize


# 求正态分布
def calcProbality(x, mean, stev):
    exponent = math.exp(-(math.pow(x - mean, 2)) / (2 * math.pow(stev, 2)))
    return (1 / (math.sqrt(2 * math.pi) * stev)) * exponent


# 在summaries中挑选最符合inputVector的概率
def calcClassProbalilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calcProbality(x, mean, stdev)
    return probabilities


# 训练数据集按照类别进行划分，然后计算每个属性的平均值和标准方差
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instance in separated.iteritems():
        summaries[classValue] = summarize(instance)
    return summaries


def predict(summaries, inputVector):
    probablilities = calcClassProbalilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probablility in probablilities.iteritems():
        if bestLabel is None or probablility > bestProb:
            bestProb = probablility
            bestLabel = classValue
    return bestLabel


def gePredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.00


if __name__ == '__main__':
    # dataset = loadCsv('test.data')
    # train, test = splitDataset(dataset, 0.87)
    # separated = separateByClass(test)
    # print separated
    summaries = {'A': [(1, 0.5)], 'B': [(29, 5.0)]}
    testSet = [[29, 1], [28, 9]]
    print gePredictions(summaries, testSet)
