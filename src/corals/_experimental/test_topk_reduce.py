import timeit
import numpy as np

# concatenate A LOT FASTER for large k

# # iterative faster
# n_arrays = 100
# size = 20000
# k = 100

# # iterative faster
# n_arrays = 20000
# size = 100
# k = 100

# # concatenate faster
# n_arrays = 200
# size = 10000
# k = 100000

# concatenate faster
n_arrays = 20000
size = 100
k = 100000

# n_arrays = 200
# size = 100
# k = 100

results = [(np.random.random(size), (np.random.random(size), np.random.random(size))) for i in range(n_arrays)]

print("running")
print("* concatenate")
from corals.correlation.topk.batched.reduce import batch_reduce_concatenate as reduce
t1 = timeit.timeit(lambda: reduce(results, k=k, threshold=None, argtopk_method="argpartition", require_sorted_topk=True), number=10)
print(t1)

# print("* iterative")
# from corals.correlation.topk_split_generic import batch_reduce_iterative as reduce
# t2 = timeit.timeit(lambda: reduce(results, k=k, threshold=None), number=10)
# print(t2)

# print("* iterative argpartition")
# from corals.correlation.topk_split_generic import batch_reduce_iterative_argpartition as reduce
# t3 = timeit.timeit(lambda: reduce(results, k=k, threshold=None), number=10)
# print(t3)

print("* iterative chunks")
from corals.correlation.topk.batched.reduce import batch_reduce_iterative_concatenate as reduce
t4 = timeit.timeit(lambda: reduce(results, k=k, threshold=None, chunk_n_batches=n_arrays), number=10)
print(t4)

print("* iterative chunks (n=10)")
from corals.correlation.topk.batched.reduce import batch_reduce_iterative_concatenate as reduce
t5 = timeit.timeit(lambda: reduce(results, k=k, threshold=None, chunk_n_batches=int(n_arrays / 100)), number=10)
print(t5)

