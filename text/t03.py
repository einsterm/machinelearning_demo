# -*- encoding:utf-8 -*-
import sys
import os
import jieba
import cPickle as pk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB

reload(sys)
sys.setdefaultencoding('utf-8')


def saveFile(savePath, content):
    f = open(savePath, "wb")
    f.write(content)
    f.close()


def readFile(filePath):
    f = open(filePath, "rb")
    content = f.read()
    f.close()
    return content


def fenci(type):
    path1 = "F:/python_wks/text_clustering_small/" + type + "/no/"
    path2 = "F:/python_wks/text_clustering_small/" + type + "/yes/"
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
            seg = jieba.cut(content)
            saveFile(savePath, " ".join(seg))


def bunchOp(type):
    fenciPath = "F:/python_wks/text_clustering_small/" + type + "/yes/"
    bunchPath = "F:/python_wks/text_clustering_small/" + type + "/bunch/" + type + "_set.dat"
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    categoryList = os.listdir(fenciPath)  # 分词后文件夹里所有的内容
    bunch.target_name.extend(categoryList)
    for categoryDirName in categoryList:
        categoryDir = fenciPath + categoryDirName
        fileList = os.listdir(categoryDir)
        for fileName in fileList:
            filePath = categoryDir + "/" + fileName
            bunch.label.append(categoryDirName)
            bunch.filenames.append(filePath)
            bunch.contents.append(readFile(filePath).strip())
    bunchFileObj = open(bunchPath, "wb")
    pk.dump(bunch, bunchFileObj)
    bunchFileObj.close()


def readBunch(filePath):
    f = open(filePath, "rb")
    bunch = pk.load(f)
    f.close()
    return bunch


def writeBunch(savePath, bunchObj):
    f = open(savePath, "wb")
    pk.dump(bunchObj, f)
    f.close()


def trainSpaceOp():
    stopWordsPath = "F:/python_wks/text_clustering_small/train/stopWords/stop.txt"
    bunchObjPath = "F:/python_wks/text_clustering_small/train/bunch/train_set.dat"
    spacePath = "F:/python_wks/text_clustering_small/train/bunch/tfidfspace.dat"
    stopWordList = readFile(stopWordsPath).splitlines()  # 停用词列表
    trainBunch = readBunch(bunchObjPath)
    trainSpace = Bunch(target_name=trainBunch.target_name, label=trainBunch.label, filenames=trainBunch.filenames,
                       tdm=[],
                       vocabulary={})
    vectorizer = TfidfVectorizer(stop_words=stopWordList, sublinear_tf=True, max_df=0.5)
    transformer = TfidfTransformer()
    trainSpace.tdm = vectorizer.fit_transform(trainBunch.contents)
    trainSpace.vocaluary = vectorizer.vocabulary_
    writeBunch(spacePath, trainSpace)


def testSpaceOp():
    stopWordsPath = "F:/python_wks/text_clustering_small/test/stopWords/stop.txt"
    testBunchObjPath = "F:/python_wks/text_clustering_small/test/bunch/test_set.dat"
    trainSpacePath = "F:/python_wks/text_clustering_small/train/bunch/tfidfspace.dat"
    testSpacePath = "F:/python_wks/text_clustering_small/test/bunch/testSpace.dat"
    stopWordsList = readFile(stopWordsPath).splitlines()

    testBunch = readBunch(testBunchObjPath)

    testSpace = Bunch(target_name=testBunch.target_name, label=testBunch.label, filenames=testBunch.filenames, tdm=[],
                      vocabulary={})

    trainBunch = readBunch(trainSpacePath)

    vectorizer = TfidfVectorizer(stop_words=stopWordsList, sublinear_tf=True, max_df=0.5,
                                 vocabulary=trainBunch.vocaluary)
    transformer = TfidfTransformer()

    testSpace.tdm = vectorizer.fit_transform(testBunch.contents)
    testSpace.vocaluary = trainBunch.vocaluary

    writeBunch(testSpacePath, testSpace)


def yuche():
    trainBunchPath = "F:/python_wks/text_clustering_small/train/bunch/tfidfspace.dat"
    testSpacePath = "F:/python_wks/text_clustering_small/test/bunch/testSpace.dat"
    trainBunch = readBunch(trainBunchPath)
    testSpace = readBunch(testSpacePath)
    clf = MultinomialNB(alpha=0.001).fit(trainBunch.tdm, trainBunch.label)
    predicted = clf.predict(testSpace.tdm)
    total = len(predicted)
    rate = 0
    for flabel, file_name, expct_cate in zip(testSpace.label, testSpace.filenames, predicted):
        if flabel != expct_cate:
            rate += 1
            print file_name
            print flabel
            print expct_cate


if __name__ == '__main__':
    # fenci("train")
    # bunchOp("train")
    #  trainSpaceOp()
    # fenci("test")
    # bunchOp("test")
    # testSpaceOp()
    yuche()
