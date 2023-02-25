import numpy as np

from corals.correlation.topk.batched.base import TopkMapReduce, ResultHandler
from corals.correlation.utils import derive_bins, argtopk


class FunctionWrapperTopkReduce(TopkMapReduce):

    def __init__(self, reduce_function, reduce_function_kwargs=None, **kwargs) -> None:
        self.reduce_function = reduce_function
        self.reduce_function_kwargs = reduce_function_kwargs

    def reduce(
            self, results,
            threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        
        # derive kwargs
        reduce_function_kwargs = {} if self.reduce_function_kwargs is None else self.reduce_function_kwargs

        # reduce
        return self.reduce_function(
            results, 
            threshold=threshold, 
            k=k, 
            result_handler=result_handler, 
            **reduce_function_kwargs)


class ProxyTopkReduce(TopkMapReduce):

    def __init__(self, reducer: TopkMapReduce=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reducer = reducer

    def reduce(
            self, results,
            threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):

        if self.reducer is None:
            reducer = FunctionWrapperTopkReduce(batch_reduce_iterative_concatenate)
        else:
            reducer = self.reducer
        
        return reducer.reduce(
            results, 
            threshold=threshold, k=k,
            n_batches=n_batches,
            approximation_factor=approximation_factor, 
            result_handler=result_handler)


def batch_reduce_merge_sorted_batches(
        result_pointers, 
        threshold, 
        k, 
        result_handler: ResultHandler=None,
        require_sorted_topk=True,
        #
        results_n_max_open=None,
        results_n_values=None,
        return_values="pointer",
        output_file=None):
    """
    TODO: implement
        * only load a couple of results?
        * only load x values from each result?
        * always write to file?
        * should probably be based on memmaps
    """

    pass



def batch_reduce_iterative_concatenate(
        result_pointers, threshold, k, 
        result_handler: ResultHandler=None,
        #
        chunk_n_batches=None, n_chunks=None, 
        argtopk_method="argpartition", 
        require_sorted_topk=True):
    """
    TODO: this COULD be parallelized through a recursive procedure 
            BUT I am not sure how much that will get us with the increased communication overhead 
            AND would require more refined memory management (more data loaded into memory simultaneously or less chunks per job.
            A dask variant may be more appropriate.
    """

    # derive `chunk_n_batches`
    if n_chunks is not None and chunk_n_batches is not None:
        raise ValueError(f"Either `chunk_n_batches` or `n_chunks` must be None.")
    elif n_chunks is None and chunk_n_batches is None:
        n_chunks = 1
    elif chunk_n_batches is not None:
        n_chunks = int(len(result_pointers) / chunk_n_batches)

    if n_chunks == 1:
        if result_handler is not None:
            result_pointers = [result_handler.load_and_remove(r) for r in result_pointers]
        result = batch_reduce_concatenate(
            result_pointers, threshold, k, argtopk_method, require_sorted_topk=require_sorted_topk)
    else:

        bins = derive_bins(len(result_pointers), n_chunks)
        
        result = []
        for i_bin in range(len(bins) - 1):
            
            if result_handler is not None:
                results_to_combine = result + [
                    result_handler.load_and_remove(r) 
                    for r in result_pointers[bins[i_bin]:bins[i_bin + 1]]]
            else:
                results_to_combine = result + result_pointers[bins[i_bin]:bins[i_bin + 1]]

            result = [batch_reduce_concatenate(
                results_to_combine, threshold, k, argtopk_method, 
                require_sorted_topk=False)]
        result = result[0]

        if require_sorted_topk:
            cor, (idx_r, idx_c) = result
            idx_sorted = np.argsort(-np.abs(cor))
            result = cor[idx_sorted], (idx_r[idx_sorted], idx_c[idx_sorted])

    return result


def batch_reduce_concatenate(
        results, 
        threshold, k, 
        argtopk_method="argpartition", 
        require_sorted_topk=True):

    topk_cor = np.concatenate([c for c, _ in results])
    topk_idx_rows = np.concatenate([i for _, (i, _) in results])
    topk_idx_cols = np.concatenate([i for _, (_, i) in results])

    # TODO: we can probably drop this argtopk implementation and replace this while thing with a numba implemnation 
    topk_idx = argtopk(
        -np.abs(topk_cor),
        k=k,
        threshold=-threshold if threshold is not None else None,
        argtopk_method=argtopk_method,
        require_sorted_topk=require_sorted_topk,
        return_values=False)

    return topk_cor[topk_idx], (topk_idx_rows[topk_idx], topk_idx_cols[topk_idx])


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
