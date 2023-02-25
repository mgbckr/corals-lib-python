def topk(
        X, Y=None, 
        threshold=None,
        k=None, 
        correlation_function="pearson", 
        **kwargs):
    raise NotImplementedError("Interface specification only.")


# convenience import for end user
# TODO: eventually replace this by more recent implementation
from ._deprecated.original import topk_balltree_combined_tree_parallel_optimized as cor_topk
