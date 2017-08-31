# -*- encoding=utf-8 -*-
import csv
import random
import math

'''
各字段含义
1.怀孕次数 
2.口服葡萄糖耐量试验中血浆葡萄糖浓度 2小时 
3.舒张压（mm Hg） 
4.三头肌皮肤褶皱厚度（mm） 
5. 2小时血清胰岛素（μU/ ml ） 
6.体重指数（kg /（身高/米）^ 2） 
7.糖尿病谱系功能 
8.年龄（岁） 
9.类变量（0或1） 
'''


# 载入数据文件，每一行是一个列表，列表每个元素是一个属性，最后一个属性是表示生病1，不生病0
def loadCsv(filename):
    lines = csv.reader(open(filename, 'rb'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# 把数据切分成，训练和测试二部分,splitRatio表示切分比例
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# 按照样本中最后一个属性（-1）为类别来生成列表
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):  # -1 表示该列表的最后一个元素，也就是倒数第一个元素，如果没有，则放入一个新的字典
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# 求平均值
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# 求标准方差
def stdev(numbers):
    avg = mean(numbers)
    sumVal = sum([pow(x - avg, 2) for x in numbers])  # pow的参数2表示平方
    variance = sumVal / float(len(numbers) - 1)
    return math.sqrt(variance)  # 求平方根


# 按列抽出属性为一个数组，计算返回得到平均值和标准方差的元组
def summarize(dataset):
    summarize = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summarize[-1]  # 最后一个属性为类别，不需要计算，去除
    return summarize


# 求正态分布，核心方法 标准差能反映一个数据集的离散程度
def calcProbality(x, mean, stev):
    exponent = math.exp(-(math.pow(x - mean, 2)) / (2 * math.pow(stev, 2)))
    return (1 / (math.sqrt(2 * math.pi) * stev)) * exponent


# 在summaries(结构是:summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]} 元组=(均值，标准方差))中挑选最符合inputVector的概率
def calcClassProbalilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1  # 给个默认值
        for i in range(len(classSummaries)):  # 属性个数
            mean, stdev = classSummaries[i]  # 第i个属性
            x = inputVector[i]
            probabilities[classValue] *= calcProbality(x, mean, stdev)
    return probabilities


# 训练数据集按照类别进行划分，然后计算每个属性的平均值和标准方差
def summarizeByClass(dataset):
    separated = separateByClass(dataset)  # 按列表的最后一个元素分组，得到类似0:[[1,2,8],[9,5,7]...] 1:[[8,5,20],[44,9,1]....]
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


# 根据训练集预测 测试集 属于那个分类
def gePredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


# 对比测试数据集真实结果，和预测数据集的结果，看预测数据集的准确性
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.00


if __name__ == '__main__':
    dataset = loadCsv('test.data')
    train, test = splitDataset(dataset, 0.99)
    separated = separateByClass(train)  # 按生病与不生病划分
    summaries = summarizeByClass(train)  # 计算每个属性的均值和标准方差
    predictions = gePredictions(summaries, test)  # 预测结果
    accuracy = getAccuracy(test, predictions)  # 把预测结果与样本的实际结果对比，看准确率
    print accuracy
