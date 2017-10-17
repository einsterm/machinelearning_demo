# --*-- encoding:utf-8 --*--
'''
求多元线性方程参数
'''

from numpy import genfromtxt
from sklearn import linear_model

dataPath = r"F:\py_wks\demo07\data.csv"
deliveryData = genfromtxt(dataPath, delimiter=',')  # 读取数据文件，以逗号分隔
X = deliveryData[:, :-1]  # 相当于dataSet
Y = deliveryData[:, -1]  # 相当于labels
regr = linear_model.LinearRegression()  # 建立多元线性方程模型 f(x)= b0+b1x1+b2x2
regr.fit(X, Y)

print regr.coef_  # 得到b1,b2
print regr.intercept_  # 得到b0

# xPred = [102.0, 6.0]
# yPred = regr.predict(xPred)
# print yPred
