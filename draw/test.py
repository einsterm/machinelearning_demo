#!/usr/bin/python
# coding: utf-8
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


a = np.arange(0, 60, 10)
for x in np.nditer(a):
    print(x)  # 0 5 10 15 20 25 30 35 40 45 50 55

# 修改数组的值: nditer对象的一个可选参数op_flags,其默认值为只读,但可以设置为读写或只写模式.这将允许使用此迭代器修改数组元素
for x in np.nditer(a, op_flags=['readwrite']):
    x[...] = 2 * x;
    print(x)