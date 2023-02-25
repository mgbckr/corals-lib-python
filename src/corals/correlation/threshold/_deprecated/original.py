import joblib
import numpy as np
import sklearn.neighbors
import corals.correlation.full.matmul

from corals.correlation.utils import *


def cor_threshold_matrix_symmetrical(X, threshold=0.75, order=False, spearman=False):

    if spearman:
        X = scipy.stats.mstats.rankdata(X, axis=0)

    cor_matrix = corals.correlation.full.matmul.full_matmul_symmetrical(X)
    idx = np.nonzero(np.abs(cor_matrix) > threshold)
    cors = cor_matrix[idx]

    if order:
        cors_order = np.argsort(-np.abs(cors))
        return cors[cors_order], (idx[0][cors_order], idx[1][cors_order])
    else:
        return cors, idx


def cor_threshold_balltree_combined_tree(X, Y=None, 
        correlation_type="pearson", threshold=0.75, tree_kwargs=None, order=False):

    # convert threshold to euclidean space 
    threshold = np.sqrt(2 * (1 - threshold))

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, Y_fill_none=True)

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    tree = sklearn.neighbors.BallTree(np.concatenate([Xh, -Xh], axis=1).transpose(), **tree_kwargs)

    # run the query
    idx, dst = tree.query_radius(Yh.transpose(), threshold, return_distance=True)

    # normalize index
    idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Xh
    mask_inverse = idx_c >= Xh.shape[1]

    # fix index for correlations from -Xh
    idx_c[mask_inverse] -= Xh.shape[1]

    # calculate correlations
    cor = 1 - np.concatenate(dst)**2 / 2

    # fix correlations from -Xh
    cor[mask_inverse] *= -1

    if order:    
        cor_order = np.argsort(-np.abs(cor))
        return cor[cor_order], (idx_r[cor_order], idx_c[cor_order])
    else:
        return cor, (idx_r, idx_c)


def cor_threshold_balltree_combined_query(X, Y=None, 
        correlation_type="pearson", threshold=0.75, tree_kwargs=None, order=False):

    # convert threshold to euclidean space 
    threshold = np.sqrt(2 * (1 - threshold))

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, Y_fill_none=True)

    # build tree
    tree = sklearn.neighbors.BallTree(Xh.transpose(), **tree_kwargs)

    # run the query
    idx, dst = tree.query_radius(
        np.concatenate([Yh, -Yh], axis=1).transpose(), 
        threshold, 
        return_distance=True)

    # normalize index
    idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Yh
    mask_inverse = idx_r >= Yh.shape[1]

    # fix index for correlations from -Yh
    idx_r[mask_inverse] -= Yh.shape[1]

    # calculate correlations
    cor = 1 - np.concatenate(dst)**2 / 2

    # fix correlations from -Yh
    cor[mask_inverse] *= -1

    if order:    
        cor_order = np.argsort(-np.abs(cor))
        return cor[cor_order], (idx_r[cor_order], idx_c[cor_order])
    else:
        return cor, (idx_r, idx_c)


def cor_threshold_balltree_combined_tree_parallel(
    X, Y=None, threshold=0.75, tree_kwargs=None, sort_results=True, 
        correlation_type="pearson", n_jobs=1, n_batches=None):

    # convert threshold to euclidean space 
    threshold = np.sqrt(2 * (1 - threshold))

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    if n_jobs is None:
        n_jobs = 1

    if n_batches is None:
        n_batches = n_jobs

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, Y_fill_none=True)

    # build tree
    tree = sklearn.neighbors.BallTree(Xh.transpose(), **tree_kwargs)
    query = np.concatenate([Yh, -Yh], axis=1)

    # derive bins
    bins = derive_bins(query.shape[1], n_batches)

    # gather results
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(tree.query_radius)(
            query[:,bins[i]:bins[i+1]].transpose(), 
            r=threshold, 
            return_distance=True,
            sort_results=sort_results) 
        for i in range(n_batches))

    dst = np.concatenate([r[1] for r in results])
    idx = np.concatenate([r[0] for r in results])

    # normalize index
    # idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_r = np.repeat(np.arange(len(idx)), [len(i) for i in idx])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Yh
    mask_inverse = idx_r >= Yh.shape[1]

    # fix index for correlations from -Yh
    idx_r[mask_inverse] -= Yh.shape[1]

    # calculate correlations
    cor = 1 - np.concatenate(dst)**2 / 2

    # fix correlations from -Yh
    cor[mask_inverse] *= -1

    if sort_results:    
        cor_order = np.argsort(-np.abs(cor))
        return cor[cor_order], (idx_r[cor_order], idx_c[cor_order])
    else:
        return cor, (idx_r, idx_c)


def cor_threshold_balltree_combined_query_parallel(
        X, Y=None, threshold=0.75, 
        correlation_type="pearson",
        tree_kwargs=None, order=False, 
        n_jobs=None, n_batches=None):

    # convert threshold to euclidean space 
    threshold = np.sqrt(2 * (1 - threshold))

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    if n_jobs is None:
        n_jobs = 1

    if n_batches is None:
        n_batches = n_jobs

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, Y_fill_none=True)

    # build tree
    tree = sklearn.neighbors.BallTree(Xh.transpose(), **tree_kwargs)

    # define query
    query = np.concatenate([Yh, -Yh], axis=1)

    # derive bins
    bins = derive_bins(query.shape[1], n_batches)

    # gather results
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(tree.query_radius)(
            query[:,bins[i]:bins[i+1]].transpose(), 
            r=threshold, 
            return_distance=True) 
        for i in range(n_batches))

    dst = np.concatenate([r[1] for r in results])
    idx = np.concatenate([r[0] for r in results])

    # normalize index
    idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Yh
    mask_inverse = idx_r >= Yh.shape[1]

    # fix index for correlations from -Yh
    idx_r[mask_inverse] -= Yh.shape[1]

    # calculate correlations
    cor = 1 - np.concatenate(dst)**2 / 2

    # fix correlations from -Yh
    cor[mask_inverse] *= -1

    if order:    
        cor_order = np.argsort(-np.abs(cor))
        return cor[cor_order], (idx_r[cor_order], idx_c[cor_order])
    else:
        return cor, (idx_r, idx_c)


def cor_threshold_balltree_twice(X, Y=None, 
        correlation_type="pearson", threshold=0.75, tree_kwargs=None, order=False):

    # convert threshold to euclidean space 
    threshold = np.sqrt(2 * (1 - threshold))

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, Y_fill_none=True)

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    tree = sklearn.neighbors.BallTree(Xh.transpose(), **tree_kwargs)

    # run the query
    idx1, dst1 = tree.query_radius(Yh.transpose(), threshold, return_distance=True)
    idx2, dst2 = tree.query_radius(-Yh.transpose(), threshold, return_distance=True)

    # normalize index
    idx1_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx1)])
    idx1_c = np.concatenate(idx1)

    idx2_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx2)])
    idx2_c = np.concatenate(idx2)

    idx_r = np.concatenate([idx1_r, idx2_r])
    idx_c = np.concatenate([idx1_c, idx2_c])

    # calculate correlations
    cor = 1 - np.concatenate([*dst1, *dst2])**2 / 2
    cor[len(idx1_r):] *= -1

    if order:    
        cor_order = np.argsort(-np.abs(cor))
        return cor[cor_order], (idx_r[cor_order], idx_c[cor_order])
    else:
        return cor, (idx_r, idx_c)


def cor_threshold_balltree_positive(X, Y=None, 
        correlation_type="pearson", threshold=0.75, tree_kwargs=None, order=False):

    # convert threshold to euclidean space 
    threshold = np.sqrt(2 * (1 - threshold))

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, Y_fill_none=True)

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    tree = sklearn.neighbors.BallTree(Xh.transpose(), **tree_kwargs)

    # run the query
    idx, dst = tree.query_radius(Yh.transpose(), threshold, return_distance=True)

    # normalize index
    idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_c = np.concatenate(idx)

    # calculate correlations
    cor = 1 - np.concatenate(dst)**2 / 2

    if order:    
        cor_order = np.argsort(-np.abs(cor))
        return cor[cor_order], (idx_r[cor_order], idx_c[cor_order])
    else:
        return cor, (idx_r, idx_c)
