# --*-- encoding:utf-8 --*--

import tree_ID3


def readData(filename):
    fr = open(filename)
    lines = fr.readlines()
    dataSet = [line.strip().split('\t') for line in lines]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataSet, labels


if __name__ == "__main__":
    dataSet, labels = readData("data")
    inTree = tree_ID3.createTree(dataSet, labels)
    tree_ID3.createPlot(inTree)
