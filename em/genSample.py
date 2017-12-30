import numpy as np
import matplotlib.pyplot as plt
import os
cov1 = np.mat("13 0;0 14")
cov2 = np.mat("15 0;0 16")
mu1 = np.array([10, 15])
mu2 = np.array([21, 25])

sample = np.zeros((100, 2))
sample[:30, :] = np.random.multivariate_normal(mean=mu1, cov=cov1, size=30)
sample[30:, :] = np.random.multivariate_normal(mean=mu2, cov=cov2, size=70)
os.remove("./gmm.data")
np.savetxt("gmm.data", sample)

plt.plot(sample[:30, 0], sample[:30, 1], "bo")
plt.plot(sample[30:, 0], sample[30:, 1], "rs")
plt.show()
