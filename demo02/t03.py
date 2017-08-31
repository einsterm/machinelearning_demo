# -*- encoding:utf-8 -*-
import sys
import os
import jieba
import cPickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB

reload(sys)
sys.setdefaultencoding('utf-8')


# 写入文件
def saveFile(savePath, content):
    f = open(savePath, "wb")
    f.write(content)
    f.close()


# 读取文件
def readFile(filePath):
    f = open(filePath, "rb")
    content = f.read()
    f.close()
    return content


def readBunch(filePath):
    f = open(filePath, "rb")
    bunch = pk.load(f)
    f.close()
    return bunch


def writeBunch(savePath, bunchObj):
    f = open(savePath, "wb")
    pk.dump(bunchObj, f)
    f.close()


# 分语,type是指要放在那个文件夹，有训练数据和测试数据
def fenci(type):
    path1 = "F:/python_wks/text_clustering_small/" + type + "/no/"  # 分词前
    path2 = "F:/python_wks/text_clustering_small/" + type + "/yes/"  # 分词后
    fileList = os.listdir(path1)
    for dirName in fileList:
        fileDir = path1 + dirName
        saveDir = path2 + dirName
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        fileList = os.listdir(fileDir)
        for fileName in fileList:
            filePath = fileDir + "/" + fileName
            savePath = saveDir + "/" + fileName
            content = readFile(filePath)
            # 下面是jieba分词
            seg = jieba.cut(content)
            saveFile(savePath, " ".join(seg))


# 生成bunch数据结构文件，方便生成向量空间模型
def bunchOp(type):
    fenciPath = "F:/python_wks/text_clustering_small/" + type + "/yes/"
    bunchPath = "F:/python_wks/text_clustering_small/" + type + "/bunch/" + type + "_set.dat"
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    categoryList = os.listdir(fenciPath)  # 分词后文件夹里所有的内容
    bunch.target_name.extend(categoryList)  # 所有类别名称列表
    for categoryDirName in categoryList:
        categoryDir = fenciPath + categoryDirName
        fileList = os.listdir(categoryDir)
        for fileName in fileList:
            filePath = categoryDir + "/" + fileName
            bunch.label.append(categoryDirName)  # 类别
            bunch.filenames.append(filePath)  # 文件路径
            bunch.contents.append(readFile(filePath).strip())  # 文件内容
    bunchFileObj = open(bunchPath, "wb")
    pk.dump(bunch, bunchFileObj)
    bunchFileObj.close()


# 生成训练向量词袋
def trainSpaceOp():
    stopWordsPath = "F:/python_wks/text_clustering_small/train/stopWords/stop.txt"
    bunchObjPath = "F:/python_wks/text_clustering_small/train/bunch/train_set.dat"
    spacePath = "F:/python_wks/text_clustering_small/train/bunch/trainSpace.dat"
    stopWordList = readFile(stopWordsPath).splitlines()  # 停用词列表
    trainBunch = readBunch(bunchObjPath)
    trainSpace = Bunch(target_name=trainBunch.target_name, label=trainBunch.label, filenames=trainBunch.filenames,
                       tdm=[],
                       vocabulary={})
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    trainSpace.tdm = vectorizer.fit_transform(trainBunch.contents)  # 转换为词频矩阵
    trainSpace.vocaluary = vectorizer.vocabulary_
    writeBunch(spacePath, trainSpace)


# 生成测试向量词袋
def testSpaceOp():
    stopWordsPath = "F:/python_wks/text_clustering_small/test/stopWords/stop.txt"
    testBunchObjPath = "F:/python_wks/text_clustering_small/test/bunch/test_set.dat"
    trainSpacePath = "F:/python_wks/text_clustering_small/train/bunch/trainSpace.dat"
    testSpacePath = "F:/python_wks/text_clustering_small/test/bunch/testSpace.dat"
    stopWordsList = readFile(stopWordsPath).splitlines()

    testBunch = readBunch(testBunchObjPath)

    testSpace = Bunch(target_name=testBunch.target_name, label=testBunch.label, filenames=testBunch.filenames, tdm=[],
                      vocabulary={})

    trainBunch = readBunch(trainSpacePath)

    vectorizer = TfidfVectorizer(stop_words=stopWordsList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainBunch.vocaluary)

    testSpace.tdm = vectorizer.fit_transform(testBunch.contents)
    testSpace.vocaluary = trainBunch.vocaluary

    writeBunch(testSpacePath, testSpace)


# 预测分类
def yuche():
    trainBunchPath = "F:/python_wks/text_clustering_small/train/bunch/trainSpace.dat"
    testSpacePath = "F:/python_wks/text_clustering_small/test/bunch/testSpace.dat"
    trainBunch = readBunch(trainBunchPath)  # 训练
    testSpace = readBunch(testSpacePath)  # 测试
    clf = MultinomialNB(alpha=0.001).fit(trainBunch.tdm, trainBunch.label)  # 朴素贝叶斯算法
    predicted = clf.predict(testSpace.tdm)  # 预测分类结果
    for flabel, file_name, expct_cate in zip(testSpace.label, testSpace.filenames, predicted):
        print file_name  # 文件名
        print flabel  # 实际类别
        print expct_cate  # 预测类别


if __name__ == '__main__':
    # fenci("train")
    # bunchOp("train")
    # trainSpaceOp()
    # fenci("test")
    # bunchOp("test")
    # testSpaceOp()
    yuche()
