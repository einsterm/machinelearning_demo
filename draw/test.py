#!/usr/bin/python
# coding: utf-8

import numpy as np

a = np.mat([1, 2, 3])
b = np.mat([1, 1, 0])

print(np.dot(a.T,b))
