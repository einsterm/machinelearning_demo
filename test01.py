import numpy as np
import matplotlib as plt

my0 = np.zeros([3, 5])
# print my0

my1 = np.ones([2, 3])
# print my1

myRandom = np.random.rand(2, 2)
# print myRandom

myeye = np.eye(3)
# print myeye

myones = np.ones([3, 3])
# print myeye + myones
# print myones - myeye

mymat = np.mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print 100 * mymat
# print mymat.sum()

mymat1 = np.mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mymat2 = np.ones(3) * 2
# print np.multiply(mymat1, mymat2)

mymat3 = np.mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print np.power(mymat3, 2)

mymat4 = np.mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mymat5 = np.mat([[0], [-1], [2]])
# print mymat4 * mymat5

mymat6 = np.mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print mymat6.T
# print mymat6.T[1]
# print mymat6
# print mymat6.T
# print mymat6 > mymat6.T

mymat7 = np.mat([[1, 2], [4, 5]])
print np.linalg.det(mymat7)
