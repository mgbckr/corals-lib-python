def topkdiff(
        D1, 
        D2, 
        threshold=None,
        k=None, 
        correlation_function="pearson", 
        **kwargs):
    raise NotImplementedError("Interface specification only.")


# convenience import for end user
from .original import topkdiff_balltree_combined_tree_parallel as cor_topkdiff