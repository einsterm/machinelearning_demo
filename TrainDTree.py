# -*- encoding:utf-8 -*-

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing

'''
Description:python调用机器学习库scikit-learn的决策树算法，实现商品购买力的预测，并转化为pdf图像显示
Author:Bai Ningchao
DateTime:2016年12月24日14:08:11
Blog URL:http://www.cnblogs.com/baiboy/
'''




def trainDicisionTree(csvfileurl):
    '读取csv文件，将其特征值存储在列表featureList中，将预测的目标值存储在labelList中'

    featureList = []
    labelList = []

    # 读取商品信息
    allElectronicsData = open(csvfileurl)
    reader = csv.reader(allElectronicsData)  # 逐行读取信息
    headers = str(allElectronicsData.readline()).split(',')  # 读取信息头文件
    print(headers)

    '存储特征数列和目标数列'
    for row in reader:
        labelList.append(row[len(row) - 1])  # 读取最后一列的目标数据
        rowDict = {}  # 存放特征值的字典
        for i in range(1, len(row) - 1):
            rowDict[headers[i]] = row[i]
            # print("rowDict:",rowDict)
        featureList.append(rowDict)
    print(featureList)
    print(labelList)

    'Vetorize features:将特征值数值化'
    vec = DictVectorizer()  # 整形数字转化
    dummyX = vec.fit_transform(featureList).toarray()  # 特征值转化是整形数据

    print("dummyX: " + str(dummyX))
    print(vec.get_feature_names())

    print("labelList: " + str(labelList))

    # vectorize class labels
    lb = preprocessing.LabelBinarizer()
    dummyY = lb.fit_transform(labelList)
    print("dummyY: \n" + str(dummyY))

    '使用决策树进行分类预测处理'
    # clf = tree.DecisionTreeClassifier()
    # 自定义采用信息熵的方式确定根节点
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(dummyX, dummyY)
    print("clf: " + str(clf))

    # Visualize model
    with open("allElectronicInformationGainOri.dot", 'w') as f:
        f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
    # '1 打开cmd进入dos环境下，并进入../Tarfile/Tname.dot路径下;#2 输入dot -Tname.dot -o name.pdf命令，将dos转化为pdf格式'
    return dummyX, dummyY







    # #修改第一行数据测试模型
    # oneRowX = dummyX[0, :]
    # print("oneRowX: " + str(oneRowX))

    # #修改001 为100 即年龄从青少年改为中年，预测购买力
    # newRowX = oneRowX
    # newRowX[0] = 1
    # newRowX[2] = 0
    # print("newRowX: " + str(newRowX))

    # # '修改后数据进行预测'


def DicisionTree(dummy, newRowX):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(dummy[0], dummy[1])
    predictedY = clf.predict(newRowX)
    print("predictedY: " + str(predictedY))


if __name__ == '__main__':
    trainDicisionTree('AllElectronics.csv')