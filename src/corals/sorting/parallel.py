import joblib
import numpy as np
from corals.correlation.utils import derive_bins


def parallel_argsort(a, n_jobs=None, n_batches=None, **argsort_kwargs):
    """
    WARNING: This does not provide a speed up. The for loop to merge the sorted arrays 
    takes too long by itself. For example try:
    ```
    %%time
    n = 50000000
    for i in range(n):       # takes 2 seconds without the inner loop
        for j in range(64):
            pass
    ```
    """

    if a.ndim > 1:
        raise ValueError("Currently only supports 1d arrays.") 

    if n_jobs is None:
        n_jobs =1

    if n_batches is None:
        n_batches = n_jobs

    bins = derive_bins(a.size, n_batches)
    
    def argsort(a):
        order = np.argsort(a)
        return order, a[order]

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(argsort)(a[bins[i]:bins[i+1]], **argsort_kwargs) 
        for i in range(n_batches))
    bin_orders = [o for o, s in results]
    bin_sorted = [s for o, s in results]

    order = np.empty(a.shape, dtype=int)
    indices = np.zeros(n_batches, dtype=int)
    for i in range(order.size):

        order_idx = None
        value_idx = None
        value = None
        
        for j in range(n_batches):
            
            if indices[j] < len(bin_sorted[j]):

                v = bin_sorted[j][indices[j]]
                if order_idx is None or v < value:
                    order_idx = j
                    value_idx = bin_orders[j][indices[j]]
                    value = v

        order[i] = value_idx + bins[order_idx]
        indices[order_idx] += 1

    return order
