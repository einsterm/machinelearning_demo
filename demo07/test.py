from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r"F:\py_wks\demo07\data.csv"
deliveryData = genfromtxt(dataPath, delimiter=',')
print "data"
print deliveryData

X = deliveryData[:, :-1]
Y = deliveryData[:, -1]

print "X:"
print X
print "Y: "
print Y

regr = linear_model.LinearRegression()

regr.fit(X, Y)

print "coefficients"
print regr.coef_
print "intercept: "
print regr.intercept_

xPred = [102, 6]
yPred = regr.predict(xPred)
print "predicted y: "
print yPred
