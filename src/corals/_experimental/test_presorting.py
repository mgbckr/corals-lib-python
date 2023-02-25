# premise: check whether sorting a batch sorted arrays is faster than sorting the original array
# conclusion: 
#   * YES! It can actually be twice as fast. 
#   * This may mean that presorting in parallel may actually give us significant speedups.
#   * however there is also a trade off since sorting in general is very slow thus and we are basically sorting twice

import timeit
import numpy as np
from corals.correlation.utils import derive_bins


X = np.random.random(1000000)

t = timeit.timeit(lambda: np.argsort(X), number=10)
print(t)

n_batches = 10
bins = derive_bins(X.size, n_batches=n_batches)

for i in range(len(bins) - 1):
    X[bins[i]:bins[i+1]] = np.sort(X[bins[i]:bins[i+1]])

t = timeit.timeit(lambda: np.argsort(X), number=10)
print(t)
