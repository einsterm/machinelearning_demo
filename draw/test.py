#!/usr/bin/python
# coding: utf-8
from numpy import *
from sklearn import preprocessing
import matplotlib.pyplot as plt

d = mat([[4, 2], [3, 6], [4, 2], [5, 2]])
f = mat([[1, 2], [2, 13]])
A = mat([[1, 0], [2, -3]])

U = mat([[-0.98708746, 0.16018224], [0.16018224, 0.98708746]])
S = mat([[0.82185442, 0], [0, 3.65028154]])
V = mat([[0.81124219, 0.58471028], [-0.58471028, 0.81124219]])
print(U * S * V)
m = mean(d, axis=0)
x = d - m
covMat = cov(x, rowvar=0)
# eigVals, eigVects = linalg.eig(A)
# print eigVects
# print eigVals


# U, S, VT = linalg.svd(A)
# print(U)
# print("---S----")
# print(S)
# print("---VT----")
# print(VT)
# sigmod = diag(S)
# print(U * sigmod * VT)
