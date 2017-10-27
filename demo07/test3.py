# --*-- encoding:utf-8 --*--
'''
SVM算法
'''
import numpy as np
import pylab as pl
from sklearn import svm

np.random.seed(0)
# X = np.r_[np.random.randn(4, 2) - [2, 2], np.random.randn(4, 2) + [2, 2]]
X = np.array([[1, 1], [1, 0], [0, 2], [2, 2], [4, 3], [4, 1], [3, 4], [5, 4]])
Y = [0] * 4 + [1] * 4

clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# w_0*x+w_1*y+w_2=0
# y=-(w_0/w_1)*x-(w_2/w_1) 这里就是点斜式
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-1, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]  # clf.intercept_[0] 是截距

b = clf.support_vectors_[0]  # 支持向量机的几个点
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80)
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
pl.axis('tight')
pl.show()
