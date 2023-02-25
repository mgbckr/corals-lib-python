import sklearn.neighbors

from corals.correlation.topk.batched.base import TopkMapReduce, ResultHandler
from corals.correlation.topk.batched.nearest_neighbors import NearestNeighborTopkMapReduce, NearestNeighborJobFocusedTopkMapReduce


class BalltreeTopkMapReduce(NearestNeighborTopkMapReduce):

    def __init__(
            self, 
            dualtree=True, 
            breadth_first=False, 
            query_sort=True, 
            tree_kwargs=None,
            argtopk_method="argpartition",
            result_handler: ResultHandler=None) -> None:

        super().__init__(
            sklearn.neighbors.BallTree, 
            tree_kwargs, 
            dict(
                breadth_first=breadth_first,
                dualtree=dualtree,
                sort_results=query_sort,
            ), 
            argtopk_method=argtopk_method,
            result_handler=result_handler)


class BalltreeJobFocusedTopkMapReduce(NearestNeighborJobFocusedTopkMapReduce):

    def __init__(
            self, 
            dualtree=True, 
            breadth_first=False, 
            query_sort=True, 
            tree_kwargs=None,
            result_handler: ResultHandler=None,
            reducer: TopkMapReduce = None) -> None:

        super().__init__(
            sklearn.neighbors.BallTree, 
            tree_kwargs, 
            dict(
                breadth_first=breadth_first,
                dualtree=dualtree,
                sort_results=query_sort,
            ), 
            result_handler=result_handler,
            reducer=reducer)
