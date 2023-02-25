from corals.correlation.topk.helper import topk_full, topk_indexed
from corals.correlation.full.baselines import full_corrcoef
from corals.correlation.threshold.loop import threshold_indexed_loop


def topk_full_corrcoef(
    X, Y=None, 
    correlation_type="pearson",
    threshold=None, 
    k=None, 
    sorting="argpartition", 
    **kwargs):

    return topk_full(
        X, 
        Y=Y, 
        correlation_type=correlation_type,
        k=k, 
        threshold=threshold, 
        #
        sorting=sorting, 
        correlation_matrix_func=full_corrcoef, 
        **kwargs)


def topk_indexed_loop(
        X, Y=None, 
        correlation_type="pearson", 
        threshold=None, 
        k=None, 
        #
        sorting="argpartition", 
        **kwargs):

    return topk_indexed(
        X, 
        Y=Y, 
        correlation_type=correlation_type,
        k=k, 
        threshold=threshold, 
        #
        sorting=sorting,
        correlation_index_func=threshold_indexed_loop)
