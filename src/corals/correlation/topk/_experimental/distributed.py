"""
    First of all we can already use Dask as a joblib backend for all batched variants.
    So far those have been slow and memory intensive though.
    This may be an issue with how I configured Dask though.

    Overall, using Dask directly 
    * could be really useful if the data does not fit on the disk
    * and the dask graph approach may be able to interleave map and reduce calls which may allow for speedups


    However, I am not sure if this will be more efficient, 
    runtime AND memory wise as dask has long startup times it seems and uses a lot of memory.
    However this may be do to me not confguring Dask correctly.

    Also, we would also have to go a bit more into details about 
    chunking and different approximation and non-approximation variants. 
    The easiest would be to not care about topk in the chunks and only threshold 
    and then only get topk in the reduce function but that would mean the topk 
    all end up in memory ... and so on.

    Overall, needs more thinking and experimenting.
"""

import numpy as np
import sparse
import dask
import dask.array as da
import dask.bag as db

from corals.correlation.utils import argtopk, derive_k_per_batch, preprocess_XY, derive_k
from corals.correlation.topk.batched.reduce import batch_reduce_concatenate


def topk_dask(
        X, 
        Y=None,
        correlation_type="pearson", 
        threshold=None,
        k=0.01,
        #
        n_batches=None,
        approximation_factor=None,
        #
        argtopk_method="argpartition",
        sort_batches=False,
        sort_topk=True):
    """
    This could be really useful if the data does not fit on the disk.
    Not sure if this will be more efficient, runtime or memory wise.
    We would also have to think about chunking and different approximation 
    and full variants that need a bit more thinking. 
    The easiest would be to not care about topk in the chunks and only threshold 
    and then only get topk in the reduce function but that would mean the topk 
    all end up in memory ... and so on.
    As I said, needs more thinking. Rant over.  
    # TODO: currently this is a really bad memory HOG!
    """

    # preprocess matrices
    # TODO: implement dask version ... but this is not the main bottleneck right now
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, normalize=True, Y_fill_none=True)

    # derive actual k (which may be given as a ratio, an into or not given at all)
    k_derived = derive_k(Xh, Yh, k=k)

    # TODO: very simple splitting here; we could split a lot more effciently if we get approximation, split numbers etc. under control
    k_per_batch = derive_k_per_batch(
        Xh, Yh, 
        n_batches=n_batches, 
        k=k_derived, 
        approximation_factor=approximation_factor)
    
    Xh = da.from_array(Xh, chunks=X.shape)
    Yh = da.from_array(Yh, chunks=(Yh.shape[0], n_batches))

    # calculate correlations

    correlation_matrix = Xh.transpose() @ Yh

    def squash(block, block_info=None):
        if block_info is None:
            return np.array([])
        idx_r, idx_c = np.unravel_index(np.arange(block.size), block.shape)
        indexed = np.stack([
            block.ravel(),
            idx_r + block_info[0]["chunk-location"][0] * block.shape[0],
            idx_c + block_info[0]["chunk-location"][1] * block.shape[1]])
        return indexed
    correlation_matrix = correlation_matrix.map_blocks(
        squash, 
        chunks=(3, np.prod(correlation_matrix.chunksize)))

    # convert to bag
    # TODO: ravel was necessary and needs to be appropriately converted back to keep indices
    result = db.from_delayed(correlation_matrix.to_delayed().ravel())

    # map

    def map_func(matrix_block):
        # TODO: careful! if both matrices are chunks matrices might be smaller than expected

        cor = matrix_block[0]
        idx_topk = argtopk(
            -np.abs(cor), 
            threshold=threshold, 
            k=k_per_batch, 
            argtopk_method=argtopk_method, 
            require_sorted_topk=sort_batches, 
            return_values=False)

        idx_r, idx_c = matrix_block[1:]

        return [(cor[idx_topk], (idx_r[idx_topk], idx_c[idx_topk]))]

    # TODO: not sure if `map_partition` makes sense here
    result = result.map_partitions(map_func)

    # fold (probably optimizable by using `combine` and `split_every``)

    def fold_function(acc, result):
        merged = [acc, result]
        return batch_reduce_concatenate(
            merged,
            threshold=threshold,
            k=k_derived,
            argtopk_method=argtopk_method,
            require_sorted_topk=sort_topk)

    result = result.fold(fold_function)

    # compute
    return result.compute()


def topk_dask_simple(
        X,
        Y=None,
        correlation_type="pearson", 
        threshold=None,
        k=0.01,
        #
        chunks=None,
        #
        argtopk_method="argpartition",
        sort_topk=True):
    """
    TODO: Implement distributed reduce function as for `topk_dask`?
    NOTE: Super long start-up time; would need to look way closer at this. Then, workers just die without any load. WTF?
    """

    # preprocess matrices
    # TODO: implement dask version ... but this is not the main bottleneck right now
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, normalize=True, Y_fill_none=False)

    # derive actual k (which may be given as a ratio, an into or not given at all)
    k_derived = derive_k(Xh, Yh, k=k)

    Xh = da.from_array(Xh, chunks=chunks)
    if Yh is None:
        Yh = Xh

    # calculate correlations

    correlation_matrix = Xh.transpose() @ Yh

    def map_blocks(block):
        block[np.abs(block) < threshold] = 0
        sparse_matrix = sparse.COO.from_numpy(block)
        return sparse_matrix 

    print("compute thresholded matrix")
    thresholded_correlation_matrix = correlation_matrix.map_blocks(map_blocks).compute()
    
    # extract correlations and indices
    idx = sparse.argwhere(thresholded_correlation_matrix)
    idx_r = idx[:, 0]
    idx_c = idx[:, 1]
    cor = sparse.COO.todense(thresholded_correlation_matrix[(idx_r, idx_c)])
    result = cor, (idx_r, idx_c)

    # reduce
    result = batch_reduce_concatenate(
        [result],
        threshold=threshold,
        k=k,
        argtopk_method=argtopk_method,
        require_sorted_topk=sort_topk)

    return result


@dask.delayed(pure=True)
def _corr_on_chunked(chunk1, chunk2, corr_thresh=None):
    if corr_thresh is not None:
        cor = np.dot(chunk1, chunk2.T)
        cor[np.abs(cor) < corr_thresh] = 0
        return sparse.COO.from_numpy(cor)
    else:
        return sparse.COO.from_numpy(np.dot(chunk1, chunk2.T))


def topk_dask_delayed(
        X,
        Y=None,
        correlation_type="pearson", 
        threshold=None,
        k=0.01,
        #
        upper_matrix_only=False,
        chunksize=5000,
        argtopk_method="argpartition",
        require_sorted_topk=True):
    # Source: https://gist.github.com/twiecki/030aa8565bae9e7ca37e3e789b6eeead
    #
    # Gets the correlation of a large DataFrame, chunking the computation
    # Returns a sparse directed adjancy matrix (old->young)
    # Adapted from https://stackoverflow.com/questions/24717513/python-numpy-corrcoef-memory-error
    #
    # TODO: would need topk processing and approximation within chunks
    #       * for approximation chunking would need to be adapted
    #       * finally, we get ImportErrors when matrices are "too dense" (~no threshold)

    Xh, Yh = preprocess_XY(
        X, Y,
        correlation_type=correlation_type, normalize=True, Y_fill_none=True)

    Xh = Xh.transpose()
    Yh = Yh.transpose()

    k = derive_k(Xh, Yh, k=k)

    numrows_X = Xh.shape[0]
    numrows_Y = Yh.shape[0]

    rows = []
    for r in range(0, numrows_X, chunksize):
        cols = []
        for c in range(0, numrows_Y, chunksize):
            r1 = r + chunksize
            c1 = c + chunksize
            chunk1 = Xh[r:r1]
            chunk2 = Yh[c:c1]
            
            delayed_array = _corr_on_chunked(chunk1, chunk2, corr_thresh=threshold)
            cols.append(da.from_delayed(
                delayed_array,
                dtype='float32',
                shape=(chunksize, chunksize),
            ))
            
        rows.append(da.hstack(cols))
        
    res = da.vstack(rows).compute()

    if upper_matrix_only:
        res = sparse.triu(res, k=1)

    idx_r, idx_c = res.tocsr().nonzero()
    cor = res[(idx_r, idx_c)].todense()

    topk_idx = argtopk(
        -np.abs(cor),
        k=k,
        threshold=-threshold if threshold is not None else None,
        argtopk_method=argtopk_method,
        require_sorted_topk=require_sorted_topk,
        return_values=False)

    return cor[topk_idx], (idx_r[topk_idx], idx_c[topk_idx]) 


if __name__ == '__main__':
    
    X = np.random.random((50, 10000))
    cor = topk_dask_delayed(X, threshold=0.5)
    pass
