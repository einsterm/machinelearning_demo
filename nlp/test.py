import pylab as pl
import numpy as np
from scipy import stats

n = 3
k = np.arange(n + 1)
pcoin = stats.binom.pmf(k, n, float(2.0 / 3))
print(pcoin)
x = stats.multinomial.pmf([1, 1, 1],n=3, p=[0.3, 0.2, 0.5])
print(x)

x = stats.multinomial.pmf([1, 2, 1],n=3, p=[0.3, 0.2, 0.5])
print(x)

x = stats.multinomial.pmf([1, 2],n=3, p=[0.66666666666667, 0.33333333333333])
print(x)
# pl.stem(k, pcoin, basefmt="k-")
# pl.margins(0.1)
