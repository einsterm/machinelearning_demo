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


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):  # 函数假设样本中最后一个属性（-1）为类别值，返回一个类别值到数据样本列表的映射
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summarize = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summarize[-1]
    return summarize


if __name__ == '__main__':
    # dataset = loadCsv('test.data')
    # train, test = splitDataset(dataset, 0.87)
    # separated = separateByClass(test)
    # print separated
    dataset = [[1, 20, 0], [2, 21, 1], [3, 22, 0]]
    print summarize(dataset)
