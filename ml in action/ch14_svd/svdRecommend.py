#!/usr/bin/python
# coding: utf-8


from numpy import linalg as la
from numpy import *


def loadExData3():
    # 利用SVD提高推荐效果，菜肴矩阵
    return [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
            [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


def loadExData2():
    # 书上代码给的示例矩阵
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def loadExData1():
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [5, 5, 5, 0, 0]]
    # return [[4, 4, 0, 2, 2],
    #         [4, 0, 0, 3, 3],
    #         [4, 0, 0, 1, 1],
    #         [1, 1, 1, 2, 0],
    #         [2, 2, 2, 0, 0],
    #         [1, 1, 1, 0, 0],
    #         [5, 5, 5, 0, 0]]


def loadExData():
    # 原矩阵
    return [[0, -1.6, 0.6],
            [0, 1.2, 0.8],
            [0, 0, 0],
            [0, 0, 0]]


# 相似度计算，假定inA和inB 都是列向量
# 基于欧氏距离
def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


# pearsSim()函数会检查是否存在3个或更多的点。
# corrcoef直接计算皮尔逊相关系数，范围[-1, 1]，归一化后[0, 1]
def pearsSim(inA, inB):
    # 如果不存在，该函数返回1.0，此时两个向量完全相关。
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


# 计算余弦相似度，如果夹角为90度，相似度为0；如果两个向量的方向相同，相似度为1.0
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# 基于物品相似度的推荐引擎
def standEst(dataMat, user_rowIdx, simMeas, eq0cloumnIdx):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user_rowIdx, j]
        if userRating == 0:
            continue

        more0cloumn = (dataMat[:, eq0cloumnIdx].A > 0)  # item_cloumnIdx列所有大于0的情况
        more0cloumn_j = (dataMat[:, j].A > 0)  # j列所有大于0的情况
        more0union = logical_and(more0cloumn, more0cloumn_j)  # 同时满足item_cloumnIdx,j列都大于0的列
        more0rowIdxs = nonzero(more0union)[0]  # 满足大于0的行下标
        if len(more0rowIdxs) == 0:
            similarity = 0
        else:
            inA = dataMat[more0rowIdxs, eq0cloumnIdx]
            inB = dataMat[more0rowIdxs, j]
            similarity = simMeas(inA, inB)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def svdEst(dataMat, user, simMeas, item):
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(diag(Sigma[: 4]))  # 生成对角矩阵，另一种方法是 Sig4 = mat(eye(4) * Sigma[: 4])
    U4 = U[:, :4]  # 截取了前4列元素
    xformedItems = dataMat.T * U4 * Sig4.I  # 去噪后的数据
    return standEst(xformedItems, user, simMeas, item)


def recommend(dataMat, user_rowIdx, N=3, simMeas=cosSim, estMethod=standEst):
    eq0boolean = (dataMat[user_rowIdx, :].A == 0)  # 矩阵的第user_rowIdx行，是否等于0的情况
    eq0position = nonzero(eq0boolean)  # 返回eq0boolean里等于True的下标，对应是那行那列
    eq0cloumIdxs = eq0position[1]  # 取列的位置，即就是等于0的下标
    if len(eq0cloumIdxs) == 0:
        return 'you rated everything'
    itemScores = []  # 物品的编号和评分值
    for eq0cloumIdx in eq0cloumIdxs:
        estimatedScore = estMethod(dataMat, user_rowIdx, simMeas, eq0cloumIdx)
        itemScores.append((eq0cloumIdx, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]


def analyse_data(Sigma, loopNum=20):
    """analyse_data(分析 Sigma 的长度取值)

    Args:
        Sigma         Sigma的值
        loopNum       循环次数
    """
    # 总方差的集合（总能量值）
    Sig2 = Sigma ** 2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i + 1])
        '''
        根据自己的业务情况，就行处理，设置对应的 Singma 次数

        通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。
        '''
        print '主成分：%s, 方差占比：%s%%' % (format(i + 1, '2.0f'), format(SigmaI / SigmaSum * 100, '4.2f'))


# 图像压缩函数
# 加载并转换数据
def imgLoadData(filename):
    myl = []
    # 打开文本文件，并从文件以数组方式读入字符
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    # 矩阵调入后，就可以在屏幕上输出该矩阵
    myMat = mat(myl)
    return myMat


# 打印矩阵
def printMat(inMat, thresh=0.8):
    # 由于矩阵保护了浮点数，因此定义浅色和深色，遍历所有矩阵元素，当元素大于阀值时打印1，否则打印0
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print 1,
            else:
                print 0,
        print ''


def imgCompress(numSV=3, thresh=0.8):
    # 构建一个列表
    myMat = imgLoadData('0_5.txt')

    print "****original matrix****"
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    analyse_data(Sigma, 20)
    SigRecon = mat(eye(numSV) * Sigma[: numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]  # 分解
    print "****reconstructed matrix using %d singular values *****" % numSV
    printMat(reconMat, thresh)


if __name__ == "__main__":
    DataMat = loadExData()
    U, Sigma, VT = linalg.svd(DataMat)
    Sig2 = mat([[Sigma[0], 0], [0, Sigma[1]]])  # 去除0的那个特征值，保留主要信息，也可以用diag函数构建对角线矩阵
    _DataMat = U[:, :2] * Sig2 * VT[:2, :]  # 截取后2，后前2 这里发现值不变，去掉了0那个特征值效果是一样的，相当于去噪点

    DataMat = mat(DataMat)
    in1 = DataMat[:, 0]  # 第一列
    in2 = DataMat[:, 1]
    in3 = DataMat[:, 2]
    d1 = ecludSim(in1, in3)
    d2 = pearsSim(in2, in3)
    d3 = cosSim(in2, in3)

    mymat = mat(loadExData1())
    # res = recommend(mymat, 2)

    mymat3 = mat(loadExData3())
    U, Simgma, VT = la.svd(mymat3)
    sig2 = Simgma ** 2

    # res = recommend(mymat3, 2, estMethod=svdEst)
    imgCompress(2)
