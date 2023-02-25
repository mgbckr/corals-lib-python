import warnings
import joblib

import numpy as np

from corals.correlation.utils import argtopk, preprocess_XY, derive_k, derive_k_per_batch, derive_bins, argsort_topk
from corals.correlation.topk.helper import topk_indexed


def _topk_split_batch(X, Y, threshold, k, sorting, offset):

    # TODO: allow to filter redundance; this might be problematic when using an approximation scheme

    cor = np.matmul(X.transpose(), Y).flatten()
    k = min(cor.size, k)  # TODO: not sure if this should be done somewhere else!?
    _, idx = argtopk(-np.abs(cor), k=k, threshold=threshold, argtopk_method=sorting)
    idx_r, idx_c = np.unravel_index(idx, (X.shape[1], Y.shape[1]))

    return cor[idx], idx_r, idx_c + offset


def _topk_split(
        X, Y=None,
        correlation_type="pearson", 
        threshold=None,
        k=None,
        #
        sorting="argpartition", 
        max_n_values=None, 
        n_batches=None,
        n_jobs=1,
        approximation_factor=10):

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, normalize=True, Y_fill_none=True)

    # derive n_batches
    if max_n_values is None and n_batches is None:
        # we assume everything fits into memory
        n_batches = 1

    elif max_n_values is not None and n_batches is None:

        # calculate `n_values_per_job` assuming that all jobs are running at the same time
        # note that this does not take into account the number of returned 
        n_values_per_job = min(max_n_values, Xh.shape[1] * Yh.shape[1]) / n_jobs
        n_batches = int(Xh.shape[1] * Yh.shape[1] / n_values_per_job)
    
    elif max_n_values is None and n_batches is not None:
        pass

    elif max_n_values is not None and n_batches is None:
        raise ValueError("Either `max_n_values` or `n_batches` can be defined.")

    if n_batches == 1:
        # we don't need to approximate when we everything it fit into one batch
        warnings.warn("Everything fit into one batch. Parallelization could be achieved using BLAS threading.")
        approximation_factor = 1

        # TODO: check whether we can keep it from starting a thread which may save memory
        n_jobs = 1

    # print(n_batches, n_jobs)

    # ks
    k_derived = derive_k(Xh, Yh, k=k)
    k_per_batch = derive_k_per_batch(Xh, Yh, n_batches, k_derived, approximation_factor)

    # define bins
    bins = derive_bins(Yh.shape[1], n_batches)

    # gather results
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_topk_split_batch)(
            Xh,
            Yh[:, bins[i]:bins[i + 1]], 
            #
            threshold=threshold,
            k=k_per_batch,
            #
            sorting=sorting,
            offset=bins[i]) 
        for i in range(n_batches))

    cor = np.concatenate([c for c, _, _ in results])
    row_idx = np.concatenate([ir for _, ir, _ in results])
    col_idx = np.concatenate([ic for _, _, ic in results])

    return cor, (row_idx, col_idx)


def topk_batch(
        X, Y=None,
        correlation_type="pearson", 
        threshold=None,
        k=None,  
        #
        sorting="argpartition", 
        max_n_values=None, 
        n_batches=None,
        n_jobs=1,
        approximation_factor=10):

    return topk_indexed(
        X=X, Y=Y,
        correlation_type=correlation_type, 
        threshold=threshold,
        k=k,  
        #
        sorting=sorting, 
        correlation_index_func=_topk_split,
        correlation_index_func_kwargs=dict(
            sorting=sorting, 
            max_n_values=max_n_values, 
            n_batches=n_batches,
            n_jobs=n_jobs,
            approximation_factor=approximation_factor))

