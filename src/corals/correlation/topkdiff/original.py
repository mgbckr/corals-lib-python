import numpy as np
import scipy.stats.mstats
import joblib
from scipy.stats.stats import spearmanr
import sklearn.neighbors
from corals.correlation.full.matmul import full_matmul_symmetrical
from corals.correlation.utils import derive_k, derive_k_per_query, preprocess_XY, argtopk
from corals.sorting.parallel import parallel_argsort

from corals.correlation.utils_numba import sort_topk_idx, symmetrize_topk
from corals.correlation.utils import derive_bins


def topkdiff_balltree_combined_tree_parallel(
        D1, D2, k=None, threshold=None, approximation_factor=10, 
        tree_kwargs=None, 
        dualtree=True, 
        breadth_first=False, 
        correlation_type="pearson",
        symmetrize=False,
        n_jobs=None, n_batches=None, 
        argtopk_method="argsort",
        require_sorted_topk=True):

    if n_jobs is None:
        n_jobs = 1

    if n_batches is None:
        n_batches = n_jobs

    if isinstance(D1, tuple):
        X1, Y1 = D1
    else:
        X1, Y1 = D1, None

    if isinstance(D2, tuple):
        X2, Y2 = D2
    else:
        X2, Y2 = D2, None

    # preprocess matrices
    X1h, Y1h = preprocess_XY(X1, Y1, correlation_type=correlation_type, normalize=True, Y_fill_none=True)
    X2h, Y2h = preprocess_XY(X2, Y2, correlation_type=correlation_type, normalize=True, Y_fill_none=True)

    k = derive_k(X1h, Y1h, k=k)
    kk = derive_k_per_query(X1h, Y1h, k, approximation_factor)

    stacked_tree = np.concatenate([X1h, X2h], axis=0)
    stacked_query = np.concatenate([Y1h, -Y2h], axis=0)

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    tree = sklearn.neighbors.BallTree(np.concatenate([stacked_tree, -stacked_tree], axis=1).transpose(), **tree_kwargs)

    # define bins
    n_batches = min(n_batches, Y1h.shape[1])
    bins = derive_bins(Y1h.shape[1], n_batches)

    # gather results
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(tree.query)(
            stacked_query[:, bins[i]:bins[i+1]].transpose(), 
            k=kk, 
            return_distance=True, 
            dualtree=dualtree, 
            breadth_first=breadth_first) 
        for i in range(n_batches))

    dst = np.concatenate([r[0] for r in results])
    idx = np.concatenate([r[1] for r in results])

    # normalize index
    idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Xh
    mask_inverse = idx_c >= stacked_tree.shape[1]

    # fix index for correlations from -Xh
    idx_c[mask_inverse] -= stacked_tree.shape[1]

    values, idx = derive_topk(
        mask_inverse, 
        np.concatenate(dst), 
        idx_r, idx_c, 
        k, threshold,
        argtopk_method=argtopk_method,
        require_sorted_topk=require_sorted_topk)

    # symmetrze to identify all topk values
    # TODO: it may make sense to only return the upper or lower triangle matrix in the first place; might be more efficient
    #       Actually I tdon't think that's possible. You will still have to sort the indices (see below)
    #       since the tree search returns values from the upper AND lower triangle.
    #       Thus, the symmetrization could be replaced by a "triangulization" which is similar to symmetrize
    #       but only returns the upper/lower triangle.
    if symmetrize:
        
        if isinstance(D1, tuple) or isinstance(D2, tuple):
            raise ValueError("Symmetrization currently only supported for one set of features (as opposed to two).")
            # TODO: Implement this; outline: 
            #   1) run top-k search again (search(Y,X) in addition to search (X,Y))
            #   2) merge the results from both
            #   This obviously takes twice the amount of time. This may not be worth it in some cases.
            #   A top-k merging procedure would be really helpful for the merge.

        values, idx = symmetrize_topk(*sort_topk_idx(values, idx))

    return values, idx


def topkdiff_matrix(
        D1, D2, k=None, correlation_type="pearson", sort="default", n_jobs_sort=1):

    if correlation_type == "spearman":
        spearman = True
    elif correlation_type == "pearson":
        spearman = False
    else:
        raise ValueError("Correlation type must be 'pearson' or 'spearman'.")

    if isinstance(D1, tuple):
        if spearman:
            D1 = scipy.stats.mstats.rankdata(D1[0], axis=0), scipy.stats.mstats.rankdata(D1[1], axis=0)
        X1, Y1 = D1
    else:
        if spearman:
            D1 = scipy.stats.mstats.rankdata(D1, axis=0)
        X1, Y1 = D1, D1

    if isinstance(D2, tuple):
        if spearman:
            D2 = scipy.stats.mstats.rankdata(D2[0], axis=0), scipy.stats.mstats.rankdata(D2[1], axis=0)
        X2, Y2 = D2
    else:
        if spearman:
            D2 = scipy.stats.mstats.rankdata(D2, axis=0)
        X2, Y2 = D2, D2

    if spearman:
        X1 = scipy.stats.mstats.rankdata(X1, axis=0)
        X2 = scipy.stats.mstats.rankdata(X2, axis=0)
        if Y1 is not None:
            Y1 = scipy.stats.mstats.rankdata(Y1, axis=0)
        if Y2 is not None:
            Y2 = scipy.stats.mstats.rankdata(Y2, axis=0)

    k = derive_k(X1, Y1, k=k)
    diff = (full_matmul_symmetrical(X1, Y1) - full_matmul_symmetrical(X2, Y2)).flatten()
    
    print("sorting", diff.shape)
    if sort == "default":
        topk_diff_order = np.argsort(-np.abs(diff))[:k]
    elif sort == "parallel-loop":
        topk_diff_order = parallel_argsort(-np.abs(diff), k=k, n_jobs=n_jobs_sort, heap=False)
    elif sort == "parallel-heap":
        topk_diff_order = parallel_argsort(-np.abs(diff), k=k, n_jobs=n_jobs_sort, heap=True)
    else:
        raise ValueError(f"Unknown search algorithm: {sort}")

    topk_diff_values = diff[topk_diff_order]

    return topk_diff_values, np.unravel_index(topk_diff_order, (X1.shape[1], Y1.shape[1]))


def topkdiff_matrix_one(
        D1, D2, k=None, correlation_type="pearson", sort="default", n_jobs_sort=1):

    if correlation_type == "spearman":
        spearman = True
    elif correlation_type == "pearson":
        pass
    else:
        raise ValueError("Correlation type must be 'pearson' or 'spearman'.")

    if isinstance(D1, tuple):
        if spearman:
            D1 = scipy.stats.mstats.rankdata(D1[0], axis=0), scipy.stats.mstats.rankdata(D1[1], axis=0)
        X1, Y1 = D1
    else:
        if spearman:
            D1 = scipy.stats.mstats.rankdata(D1, axis=0)
        X1, Y1 = D1, D1

    if isinstance(D2, tuple):
        if spearman:
            D2 = scipy.stats.mstats.rankdata(D2[0], axis=0), scipy.stats.mstats.rankdata(D2[1], axis=0)
        X2, Y2 = D2
    else:
        if spearman:
            D2 = scipy.stats.mstats.rankdata(D2, axis=0)
        X2, Y2 = D2, D2

    if spearman:
        X1 = scipy.stats.mstats.rankdata(X1, axis=0)
        X2 = scipy.stats.mstats.rankdata(X2, axis=0)
        if Y1 is not None:
            Y1 = scipy.stats.mstats.rankdata(Y1, axis=0)
        if Y2 is not None:
            Y2 = scipy.stats.mstats.rankdata(Y2, axis=0)

    k = derive_k(X1, Y1, k=k)

    X1h, Y1h = preprocess_XY(X1, Y1)
    X2h, Y2h = preprocess_XY(X2, Y2)

    stacked_X = np.concatenate([X1h, X2h], axis=0)
    stacked_Y = np.concatenate([Y1h, -Y2h], axis=0)

    diff = (stacked_X.transpose() @ stacked_Y).flatten()
    
    print("sorting", diff.shape)
    if sort == "default":
        topk_diff_order = np.argsort(-np.abs(diff))[:k]
    elif sort == "parallel-loop":
        topk_diff_order = parallel_argsort(-np.abs(diff), k=k, n_jobs=n_jobs_sort, heap=False)
    elif sort == "parallel-heap":
        topk_diff_order = parallel_argsort(-np.abs(diff), k=k, n_jobs=n_jobs_sort, heap=True)
    else:
        raise ValueError(f"Unknown search algorithm: {sort}")

    topk_diff_values = diff[topk_diff_order]

    return topk_diff_values, np.unravel_index(topk_diff_order, (X1.shape[1], Y1.shape[1]))


def derive_topk(
    mask_inverse, dst, idx_r, idx_c, k, threshold=None, 
        argtopk_method="argsort",
        require_sorted_topk=True):
    # calculate correlations

    # compared to simple top-k we need to add another 1 for top-k differences
    # since we deal with a subtraction of *two* correlations
    cor = 2 - dst**2 / 2  
    
    # # sort
    # cor_order = np.argsort(-cor)
    
    # sort
    cor_order = argtopk(
        -cor,
        threshold=-threshold if threshold is not None else None, 
        k=k, 
        argtopk_method=argtopk_method, 
        require_sorted_topk=require_sorted_topk, 
        return_values=False)

    # fix correlations from -Xh
    if mask_inverse is not None:
        cor[mask_inverse] *= -1

    return cor[cor_order][:k], (idx_r[cor_order][:k], idx_c[cor_order][:k])