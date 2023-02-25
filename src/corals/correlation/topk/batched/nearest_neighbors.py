import numpy as np

from corals.correlation.topk.batched.base import TopkMapReduce, ResultHandler
from corals.correlation.topk.batched.reduce import ProxyTopkReduce
from corals.correlation.utils import derive_k_per_batch, derive_k_per_query, argtopk


class NearestNeighborTopkMap(TopkMapReduce):
    """
    This allows to use any nearest neighbor algorithm to calculate top-k correlations.
    However, note that currently we assume that top-k correlations are extracted for each query individually.
    However, it may be better and more accurate if something like group queries would be possible.
    In this case `self.k_per_query` needs to be replaced.
    """

    def __init__(
            self,
            nearest_neighbors_class,
            nearest_neighbors_kwargs,
            query_kwargs,
            derive_k="per query",
            result_handler: ResultHandler=None,
            **kwargs) -> None:

        super().__init__(**kwargs)
        self.nearest_neighbors_class = nearest_neighbors_class
        self.nearest_neighbors_kwargs = nearest_neighbors_kwargs
        self.query_kwargs = query_kwargs
        self.result_handler = result_handler
        self.derive_k = derive_k

    def prepare(self, X, Y, threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        
        self.X_shape_ = X.shape
        
        # derive k for querying
        # NOTE: this assumes that each query is handled separately; 
        #   if group queries are supported this should be changed to `derive_k_per_batch` or something similar 
        if self.derive_k == "per query":
            self.k_query_ = derive_k_per_query(X, Y, k, approximation_factor)
        elif self.derive_k == "per batch":
            self.k_query_ = derive_k_per_batch(X, Y, n_batches, k, approximation_factor)
        else:
            self.k_query_ = self.derive_k(X, Y, k, approximation_factor)

        # prepare index
        self.prepare_index(X, Y, threshold, k, n_batches, approximation_factor, result_handler)

    def map(self, queries, queries_offset, threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        index = self.load_index()
        result = index.query(
            queries.transpose(),
            k=self.k_query_, 
            return_distance=True,
            **(self.query_kwargs if self.query_kwargs is not None else {}))
        return result if self.result_handler is None else self.result_handler.save(result)

    def prepare_index(self, X, Y, threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        self.nearest_neighbors_ = self.nearest_neighbors_class(
            np.concatenate([X, -X], axis=1).transpose(), 
            **(self.nearest_neighbors_kwargs if self.nearest_neighbors_kwargs is not None else {}))

    def load_index(self):
        return self.nearest_neighbors_


class NearestNeighborTopkMapReduce(NearestNeighborTopkMap):

    def __init__(
            self, 
            nearest_neighbors_class, 
            nearest_neighbors_kwargs, 
            query_kwargs,
            derive_k="per query",
            argtopk_method="argpartition",
            result_handler: ResultHandler=None) -> None:
        super().__init__(
            nearest_neighbors_class, 
            nearest_neighbors_kwargs, query_kwargs, derive_k, 
            result_handler)
        self.argtopk_method = argtopk_method

    def reduce(self, results, threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):

        if self.result_handler is not None:
            results = [self.result_handler(r) for r in results]

        # derive correlations
        cor = np.concatenate([r[0] for r in results]).flatten()
        cor = 1 - cor**2 / 2

        # normalize index
        idx = np.concatenate([r[1] for r in results])
        idx_r = np.repeat(np.arange(len(idx)), [len(i) for i in idx])
        idx_c = np.concatenate(idx)

        # get mask for selecting correlations from -Xh
        mask_inverse = idx_c >= self.X_shape_[1]

        # fix correlations from -Xh
        # TODO: previously this was done after sorting to save np.abs not sure if that makes a difference
        cor[mask_inverse] *= -1

        # fix index for correlations from -Xh
        idx_c[mask_inverse] -= self.X_shape_[1]

        topk_idx = argtopk(
            -np.abs(cor),
            k=k, 
            threshold=-threshold if threshold is not None else None, 
            argtopk_method=self.argtopk_method)
        
        # return
        return cor[topk_idx], (idx_r[topk_idx], idx_c[topk_idx])


class NearestNeighborJobFocusedTopkMap(NearestNeighborTopkMap):

    def __init__(
            self, 
            nearest_neighbors_class, 
            nearest_neighbors_kwargs, 
            query_kwargs,
            derive_k="per query",
            sort_joint_map_result=False,
            **kwargs) -> None:
        super().__init__(nearest_neighbors_class, nearest_neighbors_kwargs, query_kwargs, derive_k, **kwargs)
        self.sort_joint_map_result = sort_joint_map_result

    def map(self, queries, queries_offset, threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        
        cor, idx = super().map(queries, queries_offset, threshold, k, n_batches, approximation_factor, result_handler)

        # derive correlations
        cor = np.concatenate(cor)
        cor = 1 - cor**2 / 2

        # derive index
        idx_r = np.repeat(np.arange(len(idx)), [len(i) for i in idx])
        idx_c = np.concatenate(idx)
        idx_r += queries_offset

        # get mask for selecting correlations from -Yh
        mask_inverse = idx_c >= self.X_shape_[1]

        # fix correlations from -Xh
        cor[mask_inverse] *= -1

        # fix index for correlations from -Yh
        idx_c[mask_inverse] -= self.X_shape_[1]

        # TODO: this may reduce sorting times down stream, e.g., via k-way merge sort, but needs be evaluated
        if self.sort_joint_map_result:
            idx_sorted = np.argsort(-np.abs(cor))
            cor = [idx_sorted]
            idx_r = idx_r[idx_sorted]
            idx_c = idx_c[idx_sorted]
        
        return cor, (idx_r, idx_c)


class NearestNeighborJobFocusedTopkMapReduce(NearestNeighborJobFocusedTopkMap, ProxyTopkReduce):
    def __init__(
            self, 
            nearest_neighbors_class,
            nearest_neighbors_kwargs,
            query_kwargs,
            result_handler: ResultHandler=None,
            reducer: TopkMapReduce = None) -> None:
        
        super().__init__(
            nearest_neighbors_class=nearest_neighbors_class,
            nearest_neighbors_kwargs=nearest_neighbors_kwargs,
            query_kwargs=query_kwargs,
            result_handler=result_handler,
            reducer=reducer)