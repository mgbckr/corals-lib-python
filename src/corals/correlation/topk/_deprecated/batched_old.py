import warnings
import joblib
import scipy.stats.mstats
import numpy as np
from corals.correlation.utils import preprocess_XY, derive_k, derive_k_per_batch, derive_bins, argsort_topk


def topk_batch_old(
        X, Y=None, k=None, spearman=False, 
        sorting="partition", sorting_partition_final_sort=True, 
        max_n_values=None, 
        n_batches=None,
        n_jobs=1,
        approximation_factor=10):
    
    if spearman:
        X = scipy.stats.mstats.rankdata(X, axis=0)
        if Y is not None:
            Y = scipy.stats.mstats.rankdata(Y, axis=0)

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, normalize=True, Y_fill_none=True)

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
    k = derive_k(Xh, Yh, k=k)
    kk = derive_k_per_batch(Xh, Yh, n_batches, k, approximation_factor)

    # define bins
    bins = derive_bins(Yh.shape[1], n_batches)

    def topk_batch(A, B, k, offset):
        cor = np.matmul(A.transpose(), B).flatten()
        if sorting == "default":
            idx = np.argsort(-np.abs(cor))[:k]
        elif sorting == "partition":
            kth = min(k, len(cor) - 1)
            idx = np.argpartition(-np.abs(cor), kth=kth)[:k]
        else:
            raise ValueError(f"Unknown sorting function: {sorting}")

        idx_r, idx_c = np.unravel_index(idx, (A.shape[1], B.shape[1]))
        return cor[idx], idx_r, idx_c + offset

    # gather results
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(topk_batch)(
            Xh,
            Yh[:, bins[i]:bins[i+1]], 
            k=kk, 
            offset=bins[i]) 
        for i in range(n_batches))

    cor = np.concatenate([c for c, _, _ in results])
    idx_r = np.concatenate([ir for _, ir, _ in results])
    idx_c = np.concatenate([ic for _, _, ic in results])

    if sorting == "default":
        idx = np.argsort(-np.abs(cor))[:k]
    elif sorting == "partition":
        if sorting_partition_final_sort:
            idx = argsort_topk(-np.abs(cor), k)
        else:
            kth = min(k, len(cor) - 1)
            idx = np.argpartition(-np.abs(cor), kth=kth)[:k]
    else:
        raise ValueError(f"Unknown sorting function: {sorting}")

    # TODO: this actually needs to be symmetrized, e.g., like `topk_balltree_combined_tree_parallel`

    return cor[idx], (idx_r[idx], idx_c[idx])
