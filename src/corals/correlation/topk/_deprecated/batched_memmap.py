import warnings
import joblib
import pathlib

import numpy as np

from corals.correlation.utils import argtopk, preprocess_XY, derive_k, derive_k_per_batch, derive_bins


_topk_split_batch_memmap_dtype = [('negabscor', 'float32'), ('cor', 'float32'), ('idx_row', 'uint32'), ('idx_col', 'uint32')]


def _topk_batch_memmap(X, Y, threshold, k, sorting, offset, folder):

    # TODO: allow to filter redundance; this might be problematic when using an approximation scheme

    cor = np.matmul(X.transpose(), Y).flatten()
    k = min(cor.size, k)  # TODO: not sure if this should be done somewhere else!?
    _, idx = argtopk(-np.abs(cor), k=k, threshold=threshold, argtopk_method=sorting)
    cor = cor[idx]
    idx_r, idx_c = np.unravel_index(idx, (X.shape[1], Y.shape[1]))
    idx_c += offset

    result_file = folder + f"/batch_{offset}.memmap"

    memmap = np.memmap(result_file, dtype=_topk_split_batch_memmap_dtype, mode="w+", offset=0, shape=cor.size, order="C")
    memmap["negabscor"] = -np.abs(cor)
    memmap["cor"] = cor
    memmap["idx_row"] = idx_r
    memmap["idx_col"] = idx_c
    memmap.flush()

    return result_file, memmap.shape[0]


def topk_batch_memmap(
        X, 
        memmap_folder,
        Y=None,
        correlation_type="pearson", 
        threshold=None,
        k=None,
        #
        sorting="argpartition", 
        max_n_values=None, 
        n_batches=None,
        n_jobs=1,
        approximation_factor=10,
        #
        memmap_keep_batches=False,
        memmap_separate=True):

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
        joblib.delayed(_topk_batch_memmap)(
            Xh,
            Yh[:, bins[i]:bins[i+1]], 
            #
            threshold=threshold,
            k=k_per_batch,
            sorting=sorting,
            offset=bins[i],
            #
            folder=memmap_folder) 
        for i in range(n_batches))

    # merge memmaps
    correlations_memmap_tmp_file = pathlib.Path(memmap_folder) / "correlations_tmp.memmap"
    correlations_memmap_tmp = None

    for memmap_file, memmap_file_size in results:
        
        if correlations_memmap_tmp is None:
            correlations_memmap_tmp = np.memmap(correlations_memmap_tmp_file, dtype=_topk_split_batch_memmap_dtype, mode='w+',shape=memmap_file_size, order='C')
            correlations_memmap_tmp[:] = np.memmap(memmap_file, dtype=_topk_split_batch_memmap_dtype, mode='r', shape=memmap_file_size, order='C')
        
        else:
            previous_size = correlations_memmap_tmp.shape[0]
            new_size = previous_size + memmap_file_size
            correlations_memmap_tmp = np.memmap(correlations_memmap_tmp_file, dtype=_topk_split_batch_memmap_dtype, mode='r+',shape=new_size, order='C')
            correlations_memmap_tmp[previous_size:] = np.memmap(memmap_file, dtype=_topk_split_batch_memmap_dtype, mode='r',shape=memmap_file_size, order='C')

        if not memmap_keep_batches:
            pathlib.Path(memmap_file).unlink()

    # get topk
    k_max = min(k, correlations_memmap_tmp.shape[0])
    correlations_memmap_tmp.partition(k_max - 1, axis=0, order=["negabscor", "idx_row", "idx_col"])
    correlations_memmap_tmp = correlations_memmap_tmp[:k_max]
    correlations_memmap_tmp.sort(axis=0, order=["negabscor", "idx_row", "idx_col"])
    correlations_memmap_tmp = correlations_memmap_tmp[["cor", "idx_row", "idx_col"]]
    
    # write and return final memmap files
    if memmap_separate:
        
        correlations_memmap_file = pathlib.Path(memmap_folder) / "cor.memmap"
        idx_row_memmap_file = pathlib.Path(memmap_folder) / "idx_row.memmap"
        idx_col_memmap_file = pathlib.Path(memmap_folder) / "idx_col.memmap"
        
        correlations_memmap = np.memmap(correlations_memmap_file, dtype=_topk_split_batch_memmap_dtype[1][1], mode='w+',shape=correlations_memmap_tmp.size, order='C')
        idx_row_memmap = np.memmap(idx_row_memmap_file, dtype=_topk_split_batch_memmap_dtype[2][1], mode='w+',shape=correlations_memmap_tmp.size, order='C')
        idx_col_memmap = np.memmap(idx_col_memmap_file, dtype=_topk_split_batch_memmap_dtype[3][1], mode='w+',shape=correlations_memmap_tmp.size, order='C')
        
        correlations_memmap[:] = correlations_memmap_tmp["cor"]
        idx_row_memmap[:] = correlations_memmap_tmp["idx_row"]
        idx_col_memmap[:] = correlations_memmap_tmp["idx_col"]
        
        correlations_memmap.flush()
        idx_row_memmap.flush()
        idx_col_memmap.flush()

        # delete temporary correlation file
        del correlations_memmap_tmp
        correlations_memmap_tmp_file.unlink()

        return correlations_memmap, (idx_row_memmap, idx_col_memmap)

    else:
        correlations_memmap_file = pathlib.Path(memmap_folder) / "correlations.memmap"
        correlations_memmap = np.memmap(correlations_memmap_file, dtype=_topk_split_batch_memmap_dtype[1:], mode='w+',shape=correlations_memmap_tmp.size, order='C')
        correlations_memmap[:] = correlations_memmap_tmp
        correlations_memmap.flush()

        # delete temporary correlation file
        del correlations_memmap_tmp
        correlations_memmap_tmp_file.unlink()

        return correlations_memmap
