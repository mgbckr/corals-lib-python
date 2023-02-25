import numpy as np

from corals.correlation.utils import derive_k_per_batch, argtopk
from corals.correlation.topk.batched.base import TopkMapReduce, ResultHandler
from corals.correlation.topk.batched.reduce import ProxyTopkReduce


class MatmulTopkMap(TopkMapReduce):
    """
    NOTE: This can be run via threading since GIL is released by map operation!
    """

    def __init__(self, argtopk_method="argpartition", **kwargs) -> None:
        super().__init__(**kwargs)
        self.argtopk_method = argtopk_method

    def prepare(
            self, X, Y, 
            threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        
        self.X_ = X
        self.k_ = derive_k_per_batch(
            X, Y, 
            n_batches=n_batches, 
            k=k, 
            approximation_factor=approximation_factor)

    def map(
            self, queries, queries_offset,
            threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
            
        # TODO: allow to filter redundance; this might be problematic when using an approximation scheme

        cor = np.matmul(self.X_.transpose(), queries).flatten()
        topk_idx = argtopk(
            -np.abs(cor),
            k=self.k_, 
            threshold=-threshold if threshold is not None else None, 
            argtopk_method=self.argtopk_method)
        cor = cor[topk_idx]
        idx_r, idx_c = np.unravel_index(topk_idx, (self.X_.shape[1], queries.shape[1]))
        idx_c += queries_offset
        result = cor, (idx_r, idx_c)

        return result_handler.save(result, queries_offset) if result_handler is not None else result


class MatmulTopkMapReduce(MatmulTopkMap, ProxyTopkReduce):

    def __init__(self, argtopk_method="argpartition", reducer: TopkMapReduce=None) -> None:
        super().__init__(argtopk_method=argtopk_method, reducer=reducer)
