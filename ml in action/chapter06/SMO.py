#!/usr/bin/python
# coding:utf8


from numpy import *
import matplotlib.pyplot as plt


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, error):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = error
        self.m = shape(dataMatIn)[0]
        self.chenzi = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def calcEk(obj, k):
    fXk = float(multiply(obj.chenzi, obj.labelMat).T * (obj.X * obj.X[k, :].T)) + obj.b
    Ek = fXk - float(obj.labelMat[k])
    return Ek


def selectJ_maxE(i, obj, E_1):  # this is the second choice -heurstic, and calcs Ej

    max_2 = -1  # 选择第二个拉格朗日乘子的下标
    E_max = 0
    E_2 = 0
    obj.eCache[i] = [1, E_1]

    not0_E_idxs = nonzero(obj.eCache[:, 0].A)[0]  # 所有误差不为0的下标
    if (len(not0_E_idxs)) > 1:
        for k in not0_E_idxs:
            if k == i:
                continue
            Ek = calcEk(obj, k)
            E_temp = abs(E_1 - Ek)
            if (E_temp > E_max):
                max_2 = k
                E_max = E_temp
                E_2 = Ek
        return max_2, E_2
    else:
        j = selectJrand(i, obj.m)
        E_2 = calcEk(obj, j)
    return j, E_2


def updateEk(obj, k):
    Ek = calcEk(obj, k)
    obj.eCache[k] = [1, Ek]


def innerL(i, obj):
    E_1 = calcEk(obj, i)
    chenzi_1 = obj.chenzi[i]
    label_1 = obj.labelMat[i]

    if ((label_1 * E_1 < -obj.tol) and (chenzi_1 < obj.C)) or ((label_1 * E_1 > obj.tol) and (chenzi_1 > 0)):
        j, E_2 = selectJ_maxE(i, obj, E_1)  # 选择最大的误差对应的j进行优化。效果更明显
        label_2 = obj.labelMat[j]
        chenzi_2 = obj.chenzi[j]

        chenzi_1_old = chenzi_1.copy()
        chenzi_2_old = chenzi_2.copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if (label_1 != label_2):
            L = max(0, chenzi_2 - chenzi_1)
            H = min(obj.C, obj.C + chenzi_2 - chenzi_1)
        else:
            L = max(0, chenzi_2 + chenzi_1 - obj.C)
            H = min(obj.C, chenzi_2 + chenzi_1)
        if L == H:
            print("L==H")
            return 0

        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
        eta = 2.0 * obj.X[i, :] * obj.X[j, :].T - obj.X[i, :] * obj.X[i, :].T - obj.X[j, :] * obj.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0

        # 计算出一个新的alphas[j]值
        chenzi_2 -= label_2 * (E_1 - E_2) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        chenzi_2 = clipAlpha(chenzi_2, H, L)
        # 更新误差缓存
        updateEk(obj, j)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if (abs(chenzi_2 - chenzi_2_old) < 0.00001):
            print("j not moving enough")
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        chenzi_1 += label_2 * label_1 * (chenzi_2_old - chenzi_2)
        # 更新误差缓存
        updateEk(obj, i)
        b1 = obj.b - E_1 - label_1 * (chenzi_1 - chenzi_1_old) * obj.X[i, :] * obj.X[i, :].T - label_2 * (chenzi_2 - chenzi_2_old) * obj.X[i, :] * obj.X[j, :].T
        b2 = obj.b - E_2 - label_1 * (chenzi_1 - chenzi_1_old) * obj.X[i, :] * obj.X[j, :].T - label_2 * (chenzi_2 - chenzi_2_old) * obj.X[j, :] * obj.X[j, :].T
        if (chenzi_1 > 0) and (chenzi_1 < obj.C):
            obj.b = b1
        elif (0 < chenzi_2) and (obj.C > chenzi_2):
            obj.b = b2
        else:
            obj.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, error, maxIter):
    # 创建一个 optStruct 对象
    obj = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, error)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(obj.m):
                alphaPairsChanged += innerL(i, obj)
            iter += 1
        else:
            nonBoundIs = nonzero((obj.chenzi.A > 0) * (obj.chenzi.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, obj)
            iter += 1

        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
    return obj.b, obj.chenzi


def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def plotfig_SVM(xArr, yArr, ws, b, alphas):
    xMat = mat(xArr)
    yMat = mat(yArr)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 注意flatten的用法
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = arange(-1.0, 10.0, 0.1)

    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y = (-b - ws[0, 0] * x) / ws[1, 0]
    ax.plot(x, y)

    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(4):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()


if __name__ == "__main__":
    # 获取特征和目标变量
    dataArr, labelArr = loadDataSet('testSet4.txt')
    # b是常量值， alphas是拉格朗日乘子
    b, alphas = smoP(dataArr, labelArr, 1.6, 0.001, 40)
    ws = calcWs(alphas, dataArr, labelArr)
    plotfig_SVM(dataArr, labelArr, ws, b, alphas)
