# -*- encoding:utf-8 -*-
import sys
import os
import jieba
from sklearn.datasets.base import Bunch
import pickle

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


def fenci():
    path1 = "F:/python_wks/text_clustering_small/train/no/"
    path2 = "F:/python_wks/text_clustering_small/train/yes/"
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


def bunchOp():
    path1 = "F:/python_wks/text_clustering_small/train/yes/"
    path2 = "F:/python_wks/text_clustering_small/train/bunch/train_set.dat"
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    cateDirs = os.listdir(path1)
    bunch.target_name.extend(cateDirs)
    for cateDirName in cateDirs:
        saveDir = path1 + cateDirName
        fileList = os.listdir(saveDir)
        for fileName in fileList:
            filePath = saveDir + "/" + fileName
            bunch.label.append(cateDirName)
            bunch.filenames.append(filePath)
            bunch.contents.append(readFile(filePath).strip())
    fileObj = open(path2, "wb")
    pickle.dump(bunch, fileObj)
    fileObj.close()


if __name__ == '__main__':
    bunchOp()
