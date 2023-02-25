import numpy as np

from corals.correlation.full.matmul import full_matmul_symmetrical
from corals.correlation.utils import argtopk, derive_k


def topk_full(
        X, 
        Y=None, 
        threshold=None,
        k=None, 
        correlation_type="pearson",
        #
        sorting="argpartition", 
        correlation_matrix_func=None, 
        correlation_matrix_func_kwargs=None, 
        **kwargs):
    """
    Top-k correlation calculation based on calculating the whole correlation matrix.
    """

    if correlation_matrix_func is None:
        correlation_matrix_func = full_matmul_symmetrical
    if correlation_matrix_func_kwargs is None:
        correlation_matrix_func_kwargs = {}

    # calculate all correlations
    cor = correlation_matrix_func(X, Y, correlation_type=correlation_type, **correlation_matrix_func_kwargs)

    # derive a valid top-k
    k = derive_k(X, Y, k=k)

    # sort correlations
    cor = cor.flatten()
    
    topk_idx_flat = argtopk(-np.abs(cor), k=k, threshold=threshold, argtopk_method=sorting)   

    # derive topk correlation and index 
    topk_cor = cor[topk_idx_flat]
    topk_idx = np.unravel_index(topk_idx_flat, (X.shape[1], X.shape[1]))
        
    # return
    return topk_cor, topk_idx


def topk_indexed(
        X,
        Y=None, 
        threshold=None,
        k=None, 
        correlation_type="pearson",
        #
        sorting="argpartition",
        correlation_index_func=None,
        correlation_index_func_kwargs=None, 
        **kwargs):
    """
    Top-k correlation calculation based on calculating indexed correlations.
    """

    if correlation_index_func is None:
        correlation_index_func = full_matmul_symmetrical
    if correlation_index_func_kwargs is None:
        correlation_index_func_kwargs = {}
    
    # assume that the correlation function was passed as a callable
    cor, (idx_rows, idx_cols) = correlation_index_func(
        X, Y,
        correlation_type=correlation_type,
        threshold=threshold, 
        k=k, 
        **correlation_index_func_kwargs)

    # derive a valid top-k
    k = derive_k(X, Y, k=k)
    k = min(len(cor), k)

    # sort correlations
    topk_idx_flat = argtopk(-np.abs(cor), k=k, threshold=threshold, argtopk_method=sorting)   

    # derive topk correlation and index 
    topk_cor = cor[topk_idx_flat]
    topk_idx = idx_rows[topk_idx_flat], idx_cols[topk_idx_flat]
        
    # return
    return topk_cor, topk_idx
