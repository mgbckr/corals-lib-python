
from abc import abstractmethod
import warnings
import joblib

import numpy as np

# TODO: separate files since this may influence memory consumption

from corals.correlation.utils import \
    derive_k_per_query, \
    preprocess_XY, \
    argtopk, \
    derive_k, \
    derive_bins

import pynndescent


def topk_nndescent_direct(
        X,
        Y=None,
        correlation_type="pearson", 
        threshold=None,
        k=0.01,
        #
        approximation_factor=None,
        n_jobs=1,
        nndescent_index_kwargs=None,
        argtopk_method="argpartition",
        symmetrize=False):
    """
    TODO: something is wrong with the internal metrics (see `test_correlation_topk_general.py`)
    """

    if Y is not None:
        raise ValueError("Y not supported.")

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, normalize=True, Y_fill_none=True)

    # derive actual k (which may be given as a ratio, an into or not given at all)
    k_derived = derive_k(Xh, Yh, k=k)
    k_per_query = derive_k_per_query(Xh, Yh, k=k_derived, approximation_factor=approximation_factor)


    # build index
    if nndescent_index_kwargs is None:
        nndescent_index_kwargs = {}

    print("* building index")
    index = pynndescent.NNDescent(
        np.concatenate([X, -X], axis=1).transpose(), 
        n_neighbors=min(k_per_query, X.shape[1]),
        metric="euclidean",
        n_jobs=n_jobs, 
        **nndescent_index_kwargs)
    index.prepare()
    print("* done building index")

    idx, cor = index.neighbor_graph

    # derive correlations
    cor = cor.flatten()
    cor = 1 - cor**2 / 2

    # normalize index
    idx_r = np.repeat(np.arange(len(idx)), [len(i) for i in idx])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Xh
    mask_inverse = idx_c >= X.shape[1]

    # fix correlations from -Xh
    # TODO: previously this was done after sorting to save np.abs not sure if that makes a difference
    cor[mask_inverse] *= -1

    # fix index for correlations from -Xh
    idx_c[mask_inverse] -= X.shape[1]

    topk_idx = argtopk(
        -np.abs(cor),
        k=k, 
        threshold=-threshold if threshold is not None else None, 
        argtopk_method=argtopk_method)
    
    # return
    cor, idx = cor[topk_idx], (idx_r[topk_idx], idx_c[topk_idx])

    # symmetrze to identify all topk values
    # TODO: it may make sense to only return the upper or lower triangle matrix in the first place; might be more efficient
    #       Actually I tdon't think that's possible. You will still have to sort the indices (see below)
    #       since the tree search returns values from the upper AND lower triangle.
    #       Thus, the symmetrization could be replaced by a "triangulization" which is similar to symmetrize
    #       but only returns the upper/lower triangle.
    if symmetrize:

        # TODO: I suspect that imports are a big part of what makes parallelization use lots of memory.
        #       Thus, since I believe that `numba` has a large import stack I add the numba relevant imports here.
        from corals.correlation.utils_numba import sort_topk_idx, symmetrize_topk
        
        if Y is not None:
            raise ValueError("Symmetrization currently only supported for one set of features (as opposed to two).")
            # TODO: Implement X/Y summetrization; outline: 
            #   1) run top-k search again (search(Y,X) in addition to search (X,Y))
            #   2) merge the results from both
            #   This obviously takes twice the amount of time. This may not be worth it in some cases.
            #   A top-k merging procedure would be really helpful for the merge.

        cor, idx = symmetrize_topk(*sort_topk_idx(cor, idx))

    return cor, idx
