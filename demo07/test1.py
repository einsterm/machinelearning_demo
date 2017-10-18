# --*-- encoding:utf-8 --*--
'''
梯度下降法
'''
import numpy as np
import random


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    for i in range(0, numPoints):
        x[i][0] = i
        x[i][1] = 1
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        yTrans = np.dot(xTrans, loss)
        gradient = yTrans / m
        theta = theta - alpha * gradient
    return theta


if __name__ == '__main__':
    x, y = genData(3, 10, 3)
    y = np.array([2.5, 3.3, 4.0])
    print x
    print y
    m, n = np.shape(x)
    numIterations = 20000
    alpha = 0.06
    theta = np.ones(n)
    theta = gradientDescent(x, y, theta, alpha, m, numIterations)
    print theta
