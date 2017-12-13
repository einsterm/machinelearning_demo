import numpy as np

a = np.mat(np.array([3, 3, 0, 2, 0, -2, 1, -2])).T
b = np.array([1, 2])
f = np.array([[3, 2,1],[1,1,1],[2,2,2]])
d = np.mat(np.array([[2, 3], [1, 2], [3, 4]]))
e = np.array([True, False])
c = np.nonzero(a)
# d = a[:, 0]

aa={}
aa['albert']=100
aa['abbey']=999

bb={}
bb['allen']=888
aa.update(bb)

for i in range(1,10):
    print(i)


