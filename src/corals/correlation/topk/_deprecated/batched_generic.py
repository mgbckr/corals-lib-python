import warnings
import joblib
import pathlib

import numpy as np
import pickle

from corals.correlation.utils import argtopk, preprocess_XY, derive_k, derive_k_per_batch, derive_bins, argsort_topk


def topk_batch_generic(
        X,
        Y=None,
        correlation_type="pearson", 
        threshold=None,
        k=None,
        #
        batch_n_values=None, 
        n_batches=None,
        approximation_factor=None,
        n_jobs=1,
        preferred_backend=None,
        #
        batch_topk_method=None,
        batch_topk_method_kwargs=None,
        #
        batch_reduce_method=None,
        batch_reduce_method_kwargs=None):

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, normalize=True, Y_fill_none=True)

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

    if n_batches == 1:
        # we don't need to approximate when we everything it fit into one batch
        warnings.warn("Everything fit into one batch. Parallelization could be achieved using BLAS threading.")
        approximation_factor = None

        # TODO: check whether we can keep it from starting a thread which may save memory
        n_jobs = 1

    # print(n_batches, n_jobs)

    # ks
    k_derived = derive_k(Xh, Yh, k=k)
    k_per_batch = derive_k_per_batch(Xh, Yh, n_batches, k_derived, approximation_factor)

    # define bins
    bins = derive_bins(Yh.shape[1], n_batches)

    # gather results
    print("gather results")
    results = joblib.Parallel(n_jobs=n_jobs, prefer=preferred_backend)(
        joblib.delayed(batch_topk_method)(
            Xh,
            Yh[:, bins[i]:bins[i+1]], 
            #
            threshold=threshold,
            k=k_per_batch,
            offset=bins[i],
            #
            **batch_topk_method_kwargs) 
        for i in range(n_batches))

    # reduce batches to topk results
    print("reduce results")
    return batch_reduce_method(
        results, 
        threshold=threshold,
        k=k,
        **batch_reduce_method_kwargs)


_batch_topk_joint_return_dtype = [('cor', 'float32'), ('idx_row', 'uint32'), ('idx_col', 'uint32')]


def batch_topk(X, Y, threshold, k, offset, argtopk_method="argpartition"):

    # TODO: allow to filter redundance; this might be problematic when using an approximation scheme

    cor = np.matmul(X.transpose(), Y).flatten()
    idx = argtopk(
        -np.abs(cor), 
        k=k, 
        threshold=-threshold, 
        argtopk_method=argtopk_method)
    idx_r, idx_c = np.unravel_index(idx, (X.shape[1], Y.shape[1]))

    return cor[idx], (idx_r, idx_c + offset)


def batch_topk_memmap_joint(X, Y, threshold, k, sorting, offset, folder):

    # TODO: allow to filter redundance (only return row < col); this might be problematic when using an approximation scheme

    cor = np.matmul(X.transpose(), Y).flatten()
    idx = argtopk(-np.abs(cor), k=k, threshold=-threshold, argtopk_method=sorting, return_values=False)
    cor = cor[idx]
    idx_r, idx_c = np.unravel_index(idx, (X.shape[1], Y.shape[1]))
    idx_c += offset

    result_file = pathlib.Path(folder) /  f"batch_{offset}.memmap"

    # TODO: maybe a simple pickle is enough
    memmap = np.memmap(result_file, dtype=_batch_topk_joint_return_dtype, mode="w+", offset=0, shape=cor.size, order="C")
    memmap["cor"] = cor
    memmap["idx_row"] = idx_r
    memmap["idx_col"] = idx_c
    memmap.flush()

    return result_file, memmap.shape[0]


def batch_topk_pickle(X, Y, threshold, k, offset, out_folder, argtopk_method="argpartition"):

    # TODO: allow to filter redundance (only return row < col); this might be problematic when using an approximation scheme

    cor = np.matmul(X.transpose(), Y).flatten()
    idx = argtopk(-np.abs(cor), k=k, threshold=-threshold, argtopk_method=argtopk_method, return_values=False)
    cor = cor[idx]
    idx_r, idx_c = np.unravel_index(idx, (X.shape[1], Y.shape[1]))
    idx_c += offset

    result_file = pathlib.Path(out_folder) /  f"batch_{offset}.pickle"

    pickle.dump((cor, (idx_r, idx_c)), open(result_file, "wb"))

    return result_file, cor.size


def batch_reduce_iterative(results, threshold, k):
    """
    Way too slow for larger k. Very fast for very small k (obviously).
    TODO: maybe add a numba or cython implementation of this to make it more viable?
    """

    topk_cor = np.empty(k)
    topk_idx_rows = np.empty(k)
    topk_idx_cols = np.empty(k)

    iter_cor = [iter(c) for c, _ in results]
    iter_idx_rows = [iter(i) for _, (i, _) in results]
    iter_idx_cols = [iter(i) for _, (_, i) in results]

    current_cor = np.array([next(v) for v in iter_cor])
    current_abscor = np.abs(current_cor)
    current_idx_rows = np.array([next(v) for v in iter_idx_cols])
    current_idx_cols = np.array([next(v) for v in iter_idx_rows])

    for i in range(k):

        # print(current_abscor)
        # print(np.argmax(current_abscor))
        idx_max = np.argmax(current_abscor)
        
        # make sure the threshold is honored
        if threshold is not None and current_cor[idx_max] < threshold:
            break

        topk_cor[i] = current_cor[idx_max]
        topk_idx_rows[i] = current_idx_rows[idx_max]
        topk_idx_cols[i] = current_idx_cols[idx_max]

        current_cor[idx_max] = next(iter_cor[idx_max])
        current_abscor[idx_max] = np.abs(current_cor[idx_max])  # TODO: better to just run abs(current_cor) every time?
        current_idx_rows[idx_max] = next(iter_idx_rows[idx_max])
        current_idx_cols[idx_max] = next(iter_idx_cols[idx_max])

    return topk_cor[:i], (topk_idx_rows[:i], topk_idx_cols[:i])


def batch_reduce_iterative_argpartition(results, threshold, k,):
    """WAAAAAAY TOO SLOW"""

    # TODO: implement final sort
    # TODO: implement thresholding

    topk_cor = np.empty(0)
    topk_idx_rows = np.empty(0)
    topk_idx_cols = np.empty(0)
    for r in results:
        cor, (idx_rows, idx_cols) = r
        merged_cor = np.concatenate([cor, topk_cor])
        merged_idx_rows = np.concatenate([idx_rows, topk_idx_rows])
        merged_idx_cols = np.concatenate([idx_cols, topk_idx_cols])
        topk_idx = np.argpartition(-np.abs(merged_cor), kth=min(k - 1, len(merged_cor) - 1))
        topk_cor = merged_cor[topk_idx]
        topk_idx_rows = merged_idx_rows[topk_idx]
        topk_idx_cols = merged_idx_cols[topk_idx]

    return topk_cor, (topk_idx_rows, topk_idx_cols)


def load_result_joint_numpy(r):
    path, _ = r
    array = np.load(open(path, "rb"))
    return array["cor"], (array["idx_row"], array["idx_col"])


def load_result_pickle(r):
    path, _ = r
    return pickle.load(open(path, "rb"))


def cleanup_result_delete(r):
    path, _ = r
    pathlib.Path(path).unlink()


def batch_reduce_iterative_concatenate(
        results, threshold, k, 
        chunk_n_batches=None, n_chunks=None, 
        argtopk_method="argpartition", 
        require_sorted_topk=True,
        load_result=None,
        cleanup_result=None):
    """
    TODO: replace individual list append loop by bin slicing to get rid of the append overhead (not sure how mich that will help) 
    TODO: this COULD be parallelized through a recursive procedure 
            BUT I am not sure how much that will get us with the increased communication overhead 
            AND would require more refined memory management (more data loaded into memory simultaneously or less chunks per job.
            A dask variant may be more appropriate.
    """

    # derive `chunk_n_batches`
    if n_chunks is not None and chunk_n_batches is not None:
        raise ValueError(f"Either `chunk_n_batches` or `n_chunks` must be None.")
    elif n_chunks is None and chunk_n_batches is None:
        chunk_n_batches = len(results)
    elif n_chunks is not None:
        chunk_n_batches = int(len(results) / n_chunks)

    # combine chunks
    results_to_combine = []  
    for i, r in enumerate(results):
        
        if load_result:
            r_loaded = load_result(r)
            if cleanup_result is not None:
                cleanup_result(r)
            r = r_loaded

        results_to_combine.append(r)

        if i % chunk_n_batches == 0:
            result = batch_reduce_concatenate(
                results_to_combine, threshold, k, argtopk_method, require_sorted_topk=False)
            results_to_combine = [result]
    
    # add remaining batches
    # TODO: remove this call?
    result = batch_reduce_concatenate(
        results_to_combine, threshold, k, argtopk_method, require_sorted_topk=require_sorted_topk)

    return result


def batch_reduce_concatenate(results, threshold, k, argtopk_method="argpartition", require_sorted_topk=True):

    topk_cor = np.concatenate([c for c, _ in results])
    topk_idx_rows = np.concatenate([i for _, (i, _) in results])
    topk_idx_cols = np.concatenate([i for _, (_, i) in results])

    # TODO: we can probably drop this argtopk implementation and replace this while thing with a numba implemnation 
    topk_idx = argtopk(
        -np.abs(topk_cor),
        k=k,
        threshold=-threshold,
        argtopk_method=argtopk_method,
        require_sorted_topk=require_sorted_topk,
        return_values=False)

    return topk_cor[topk_idx], (topk_idx_rows[topk_idx], topk_idx_cols[topk_idx])
