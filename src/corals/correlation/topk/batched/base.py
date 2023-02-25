from abc import abstractmethod
import warnings
import joblib

import numpy as np
from threadpoolctl import threadpool_limits

# TODO: separate files since this may influence memory consumption

from corals.correlation.utils import \
    preprocess_XY, \
    argtopk, \
    derive_k, \
    derive_bins

joint_array_result_dtype = [('cor', 'float32'), ('idx_row', 'uint32'), ('idx_col', 'uint32')]


class ResultHandler():
    
    @abstractmethod
    def init(self):
        pass
    
    @abstractmethod
    def save(self, result, offset):
        pass
    
    @abstractmethod
    def load(self, result_pointer, **kwargs):
        pass
    
    @abstractmethod
    def remove(self, result_pointer):
        pass

    @abstractmethod
    def close(self):
        pass

    def load_and_remove(self, result_pointer, **kwargs):
        r = self.load(result_pointer, **kwargs)
        self.remove(result_pointer)
        return r

class TopkMapReduce():
    """
    TODO: evaluate whether `.map` and `.reduce` signatures should be simplified.
        This would mean storing 
            * threshold, 
            * k, 
            * n_batches, 
            * approximation_factor and 
            * result_handler
        as class variables.
        Pros:
            * cleaner calls
            * less room for error
        Cons: 
            * classes more stateful thus possibly harder to understand
            * harder to uncouple mappers and reducers
    """

    def prepare(
            self, X, Y, 
            threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        pass

    @abstractmethod
    def map(
            self, queries, queries_offset, 
            threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        pass

    @abstractmethod
    def reduce(
            self, results,
            threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        pass


def topk_batched_generic(
        X,
        Y=None,
        correlation_type="pearson", 
        threshold=None,
        k=0.01,
        #
        batch_n_values=None, 
        n_batches=None,
        approximation_factor=None,
        n_jobs=1,
        symmetrize=False,
        #
        mapreduce: TopkMapReduce=None,
        attempt_limiting_blas_threads_in_map=None,
        result_handler: ResultHandler=None, 
        **kwargs):
    """
    Chunks the query matrix and 
    calculates the top-k for each chunk separately (map) 
    before merging them into the overall result (reduce).

    TODO: Currently only the query is chunked. Allow chunking the search space as well? 
        For BallTrees this would mean to create a tree for each chunk.
        This would probably change the API a lot again, so I won't mess with it for now.
    """

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, normalize=True, Y_fill_none=True)

    # derive actual k (which may be given as a ratio, an into or not given at all)
    k_derived = derive_k(Xh, Yh, k=k)

    # derive n_batches
    if batch_n_values is None and n_batches is None:
        # we assume everything fits into memory
        n_batches = 1

    elif batch_n_values is not None and n_batches is None:

        # calculate `n_values_per_job` assuming that all jobs are running at the same time
        # note that this does not take into account the number of returned 
        n_values_per_job = min(batch_n_values, Xh.shape[1] * Yh.shape[1]) / n_jobs
        n_batches = int(Xh.shape[1] * Yh.shape[1] / n_values_per_job)
    
    elif batch_n_values is None and n_batches is not None:
        pass

    elif batch_n_values is not None and n_batches is None:
        raise ValueError("Either `max_n_values` or `n_batches` can be defined.")

    # we do not need to run in parallel when k fits into one batch
    if n_batches == 1:
        warnings.warn(
            "Everything fit into one batch. Parallelization could be achieved using BLAS threading.")
        
        if approximation_factor is not None:
            warnings.warn(
                f"Everything fit into one batch. Make sure the approximation factor needs to be set for the current setup: {approximation_factor}.")

        # this should should prevent joblib to spwan threads, theoretically saving time and memory
        n_jobs = 1

    # shared map reduce args
    mapreduce_kwargs = dict(
        threshold=threshold, k=k_derived, 
        n_batches=n_batches, 
        approximation_factor=approximation_factor, 
        result_handler=result_handler
    )

    # prepare
    print("* prepare")
    if result_handler is not None:
        result_handler.init()
    mapreduce.prepare(Xh, Yh, **mapreduce_kwargs)

    # map
    print("* map")
    if attempt_limiting_blas_threads_in_map is not None:
        # TODO: this is an attempt to fix schedulers like dask not adhering to BLAS limit set via environment variables
        #       but so far when using this dask just never finishes
        #       the same is actually true for the joblib threading backend; could this be caused by the GIL?
        def limited_map(**kwargs):
            with threadpool_limits(limits=attempt_limiting_blas_threads_in_map, user_api='blas'):
                return mapreduce.map(**kwargs)
    else:
        limited_map = mapreduce.map

    bins = derive_bins(Yh.shape[1], n_batches)
    results = joblib.Parallel(n_jobs=n_jobs, backend="spark")(
        joblib.delayed(limited_map)(
            queries=Yh[:, bins[i]:bins[i+1]],
            queries_offset=bins[i],
            **mapreduce_kwargs)
        for i in range(len(bins) - 1))

    # reduce batches to topk results
    print("* reduce")
    cor, idx = mapreduce.reduce(
        results, 
        **mapreduce_kwargs)

    if result_handler is not None:
        result_handler.close()

    # symmetrze to identify all topk values
    # TODO: it may make sense to only return the upper or lower triangle matrix in the first place; might be more efficient
    #       Actually I tdon't think that's possible. You will still have to sort the indices (see below)
    #       since the tree search returns values from the upper AND lower triangle.
    #       Thus, the symmetrization could be replaced by a "triangulization" which is similar to symmetrize
    #       but only returns the upper/lower triangle.
    if symmetrize:

        # TODO: I suspect that imports are a big part of what makes parallelization use lots of memory.
        #       Thus, since I believe that `numba` has a large import stack I add the numba relevant imports here.
        from corals.correlation.utils_numba import sort_topk_idx, symmetrize_topk
        
        if Y is not None:
            raise ValueError("Symmetrization currently only supported for one set of features (as opposed to two).")
            # TODO: Implement X/Y summetrization; outline: 
            #   1) run top-k search again (search(Y,X) in addition to search (X,Y))
            #   2) merge the results from both
            #   This obviously takes twice the amount of time. This may not be worth it in some cases.
            #   A top-k merging procedure would be really helpful for the merge.

        cor, idx = symmetrize_topk(*sort_topk_idx(cor, idx))

    return cor, idx
