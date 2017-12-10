# --*-- encoding:utf-8 --*--
from numpy import *


def loadSimpData():
    # dataArr = array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    # labelArr = [1.0, 1.0, -1.0, -1.0, 1.0]
    dataArr = array([[0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]])
    labelArr = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0]
    return dataArr, labelArr


def stumpClassify(dataMat, colIdx, compareVal, op):
    oneMat = ones((shape(dataMat)[0], 1))
    col = dataMat[:, colIdx]
    if op == 'lt':
        oneMat[col <= compareVal] = -1.0  # 使oneMat[True]的元素等于-1
    else:
        oneMat[col > compareVal] = -1.0
    return oneMat


def buildStump(dataArr, labelArr, D):
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).T
    m, n = shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestLabel_yuche = mat(zeros((m, 1)))
    minError = inf  # 无限大
    for i in range(n):
        minData = dataMat[:, i].min()
        maxData = dataMat[:, i].max()
        stepSize = (maxData - minData) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for operator in ['lt', 'gt']:
                compareVal = (minData + float(j) * stepSize)  # 拿这个值一个个去作比较
                yucheLabel = stumpClassify(dataMat, i, compareVal, operator)  # 预测值
                errArr = mat(ones((m, 1)))
                errArr[yucheLabel == labelMat] = 0  # 正确为0，错误为1
                allError = D.T * errArr  # 把本次以compareVal为分割点的分类，出现的错误分类加和
                print("本次循环以 %s 为分割点 ，错误率为 %s" % (compareVal, allError))
                if allError < minError:
                    minError = allError
                    bestLabel_yuche = yucheLabel.copy()
                    bestStump['colIdx'] = i
                    bestStump['compareVal'] = compareVal
                    bestStump['operator'] = operator
    return bestStump, minError, bestLabel_yuche


def adaBoostTrainDS(dataArr, labelArr, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 每分类错一行数据的概率
    yucheValues = mat(zeros((m, 1)))
    for i in range(numIt):
        bestResult, errorMat, yucheResult = buildStump(dataArr, labelArr, D)  # 得到一个简单的决策树预测的结果
        alpha = float(0.5 * log((1.0 - errorMat) / max(errorMat, 1e-16)))  # 1e-16是给默认值，为了使分母不为零
        bestResult['alpha'] = alpha
        weakClassArr.append(bestResult)
        expon = multiply(-1 * alpha * mat(labelArr).T, yucheResult)  # 判断正确的，就乘以-1，否则就乘以1， 为什么？ 书上的公式。
        D = multiply(D, exp(expon))
        D = D / D.sum()
        yucheValues += alpha * yucheResult
        yucheVal_sign = sign(yucheValues)  # sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
        eqVal = yucheVal_sign != mat(labelArr).T
        errorPosition = multiply(eqVal, ones((m, 1)))  # 结果为：错误的样本标签集合，因为是 !=,那么结果就是0 正, 1 负
        errorCount = errorPosition.sum()
        errorRate = errorCount / m
        print("total error=%s " % (errorRate))
        if errorRate == 0.0:
            break
    return weakClassArr, yucheValues


def adaClassify(datToClass, classifierArr):
    dataMat = mat(datToClass)
    m = shape(dataMat)[0]
    label_yuche = mat(zeros((m, 1)))
    classifierArrLen = len(classifierArr)
    for i in range(classifierArrLen):  # 循环 多个分类器
        # 前提： 我们已经知道了最佳的分类器的实例
        # 通过分类器来核算每一次的分类结果，然后通过alpha*每一次的结果 得到最后的权重加和的值。
        colIdx = classifierArr[i]['colIdx']
        compareVal = classifierArr[i]['compareVal']
        operator = classifierArr[i]['operator']
        alpha = classifierArr[i]['alpha']
        classEst = stumpClassify(dataMat, colIdx, compareVal, operator)
        label_yuche += alpha * classEst
    return sign(label_yuche)


def loadDataSet(fileName):
    # get number of fields
    numFeat = len(open(fileName).readline().split('\t'))
    dataArr = []
    labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataArr.append(lineArr)
        labelArr.append(float(curLine[-1]))
    return dataArr, labelArr


if __name__ == "__main__":
    dataArr, labelArr = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 9)
    result = adaClassify([5], weakClassArr)
    print(result)
    # dataArr, labelArr = loadDataSet("horseColicTraining2.txt")
    # weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
