import numpy as np
from scipy.stats.stats import spearmanr
import sklearn.neighbors
import joblib
import scipy.stats.mstats

from corals.correlation.utils import derive_bins
from corals.sorting.parallel import parallel_argsort

from corals.correlation.full.matmul import full_matmul_symmetrical

from corals.correlation.utils import preprocess_XY, derive_k, derive_k_per_query, derive_topk
from corals.correlation.utils_numba import sort_topk_idx, symmetrize_topk


def topk_corrcoef(X, Y=None, k=None):

    k = derive_k(X, X, k=k)

    cor = np.corrcoef(X, rowvar=False).flatten()
    topk_order = np.argsort(-np.abs(cor))[:k]
    topk_values = cor[topk_order]
        
    return topk_values, np.unravel_index(topk_order, (X.shape[1], X.shape[1]))


def topk_matrix(X, Y=None, k=None, spearman=False, sorting="default", n_jobs_sort=1):
    
    if spearman:
        X = scipy.stats.mstats.rankdata(X, axis=0)
        if Y is not None:
            Y = scipy.stats.mstats.rankdata(Y, axis=0)
            
    k = derive_k(X,Y, k=k)

    cor = full_matmul_symmetrical(X, Y, avoid_copy=True).flatten()

    if sorting == "default":
        topk_order = np.argsort(-np.abs(cor))[:k]
    elif sorting == "partition":
        topk_order = np.argpartition(-np.abs(cor), kth=(k -1))[:k]
    elif sorting == "parallel-loop":
        topk_order = parallel_argsort(-np.abs(cor), k=k, n_jobs=n_jobs_sort, heap=False)
    elif sorting == "parallel-heap":
        topk_order = parallel_argsort(-np.abs(cor), k=k, n_jobs=n_jobs_sort, heap=True)
    else:
        raise ValueError(f"Unknown search algorithm: {sorting}")

    topk_values = cor[topk_order]
        
    return topk_values, np.unravel_index(topk_order, (X.shape[1], X.shape[1] if Y is None else Y.shape[1]))


def topk_matrix_parallel_sort(X, Y=None, k=None, n_jobs=None):
    """
    WARNING: This does not provide a speed up. See `parallel_argsort` for details.
    ```
    """

    if Y is None:
        Y = X
        
    k = derive_k(X, Y, k=k)

    if n_jobs is None:
        n_jobs = 1

    cor = full_matmul_symmetrical(X, Y).flatten()
    topk_order = parallel_argsort(-np.abs(cor), n_jobs=n_jobs)
    topk_values = cor[topk_order[:k]]
        
    return topk_values, np.unravel_index(topk_order[:k], (X.shape[1], Y.shape[1]))


def topk_balltree_combined_tree(X, Y=None, correlation_type="pearson", k=None, approximation_factor=10, tree_kwargs=None, dualtree=True, breadth_first=False):
    
    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, normalize=True, correlation_type=correlation_type, Y_fill_none=True)
    k = derive_k(Xh, Yh, k=k)
    kk = derive_k_per_query(Xh, Yh, k, approximation_factor)

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    tree = sklearn.neighbors.BallTree(np.concatenate([Xh, -Xh], axis=1).transpose(), **tree_kwargs)

    # run the query
    dst, idx = tree.query(Yh.transpose(), k=kk, return_distance=True, dualtree=dualtree, breadth_first=breadth_first)

    # normalize index
    idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Xh
    mask_inverse = idx_c >= Xh.shape[1]

    # fix index for correlations from -Xh
    idx_c[mask_inverse] -= Xh.shape[1]

    return derive_topk(mask_inverse, np.concatenate(dst), idx_r, idx_c,k)


def topk_balltree_combined_tree_parallel(
        X, Y=None, correlation_type="pearson", k=None, approximation_factor=10, tree_kwargs=None, dualtree=True, breadth_first=False,
        n_jobs=None, n_batches=None, symmetrize=False, sort="default"):

    if n_jobs is None:
        n_jobs = 1

    if n_batches is None:
        n_batches = n_jobs

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, normalize=True, correlation_type=correlation_type, Y_fill_none=True)
    k = derive_k(Xh, Yh, k=k)
    kk = derive_k_per_query(Xh, Yh, k, approximation_factor)

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    tree = sklearn.neighbors.BallTree(np.concatenate([Xh, -Xh], axis=1).transpose(), **tree_kwargs)

    # define bins
    bins = derive_bins(Yh.shape[1], n_batches)

    # test memory efficiency optimzation (didn't work)
    Yht = Yh.transpose()
    def test(i):
        return tree.query(
            Yht[bins[i]:bins[i+1],:], 
            k=kk, 
            return_distance=True, 
            dualtree=dualtree, 
            breadth_first=breadth_first) 

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(test)(i)
        for i in range(n_batches))

    # gather results
    # results = joblib.Parallel(n_jobs=n_jobs)(
    #     joblib.delayed(tree.query)(
    #         Yh[:, bins[i]:bins[i+1]].transpose(), 
    #         k=kk, 
    #         return_distance=True, 
    #         dualtree=dualtree, 
    #         breadth_first=breadth_first) 
    #     for i in range(n_batches))
    # print("done")

    # TODO: 
    #   we could move the sorting into the parallel step 
    #   and then use a heap-based merge to combine these presorted arrays
    #   (see corals.sorting.sort.parallel_argsort)

    dst = np.concatenate([r[0] for r in results])
    idx = np.concatenate([r[1] for r in results])

    # normalize index
    # idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_r = np.repeat(np.arange(len(idx)), [len(i) for i in idx])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Xh
    mask_inverse = idx_c >= Xh.shape[1]

    # fix index for correlations from -Xh
    idx_c[mask_inverse] -= Xh.shape[1]

    values, idx = derive_topk(mask_inverse, np.concatenate(dst), idx_r, idx_c, k)
    # symmetrze to identify all topk values
    # TODO: it may make sense to only return the upper or lower triangle matrix in the first place; might be more efficient
    #       Actually I don't think that's possible. You will still have to sort the indices (see below)
    #       since the tree search returns values from the upper AND lower triangle.
    #       Thus, the symmetrization could be replaced by a "triangulization" which is similar to symmetrize
    #       but only returns the upper/lower triangle.
    if symmetrize:
        
        if Y is not None:
            raise ValueError("Symmetrization currently only supported for one set of features (as opposed to two).")
            # TODO: Implement this; outline: 
            #   1) run top-k search again (search(Y,X) in addition to search (X,Y))
            #   2) merge the results from both
            #   This obviously takes twice the amount of time. This may not be worth it in some cases.
            #   A top-k merging procedure would be really helpful for the merge.

        values, idx = symmetrize_topk(*sort_topk_idx(values, idx))

    return values, idx


def topk_balltree_combined_tree_parallel_optimized(
        X, 
        Y=None, 
        correlation_type="pearson",
        k=None, 
        threshold=None,
        approximation_factor=10, 
        tree_kwargs=None, 
        dualtree=True, 
        breadth_first=False,
        query_sort=True,
        n_batches=None, 
        n_jobs=None, 
        # joblib's default breaks for too much data `n_jobs_transfer_mode` implements potential fixes.
        # However, this fix didn't always work.
        # Fortunately, this issue was fixed in Python 3.9 and thus the corresponding workarounds are obsolete.
        n_jobs_transfer_mode="function",  
        symmetrize=False, 
        argtopk_method="argsort",
        require_sorted_topk=True,
        handle_zero_variance="raise" # None, "raise", "return indices", 
    ):
    
    if handle_zero_variance is not None:
        
        X = np.array(X)

        X_zero_var_msk = np.all(np.isclose(X, X[0,:]), axis=0)
        X_zero_var_idx = np.arange(X.shape[1])[X_zero_var_msk]

        if handle_zero_variance == "raise":
            if len(X_zero_var_idx) > 0:
                raise ValueError(
                    f"Zero variance in X. Please remove. Indices: {X_zero_var_idx}")
            
        if Y is not None:

            Y = np.array(Y)

            Y_zero_var_msk = np.all(np.isclose(Y, Y[0,:]), axis=0)
            Y_zero_var_idx = np.arange(Y.shape[1])[Y_zero_var_msk]

            if handle_zero_variance == "raise":
                if len(Y_zero_var_idx) > 0:
                    raise ValueError(
                        f"Zero variance in Y. Please remove. Indices: {Y_zero_var_idx}")

        if handle_zero_variance == "return indices":

            if Y is None:
                return None, X_zero_var_idx
            else:
                return None, (X_zero_var_idx, Y_zero_var_idx)


    if n_jobs is None:
        n_jobs = 1

    if n_batches is None:
        n_batches = n_jobs

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, normalize=True, Y_fill_none=True, correlation_type=correlation_type)
    k = derive_k(Xh, Yh, k=k)
    kk = derive_k_per_query(Xh, Yh, k, approximation_factor)

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    tree = sklearn.neighbors.BallTree(np.concatenate([Xh, -Xh], axis=1).transpose(), **tree_kwargs)

    # define bins
    bins = derive_bins(Yh.shape[1], n_batches)

    # gather results
    # print(f"Transfer mode: {n_jobs_transfer_mode}")
    if n_jobs_transfer_mode == "direct":

        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(tree.query)(
                Yh[:, bins[i]:bins[i+1]].transpose(), 
                k=kk, 
                return_distance=True, 
                dualtree=dualtree, 
                breadth_first=breadth_first,
                sort_results=query_sort)
            for i in range(n_batches))
    
    elif n_jobs_transfer_mode == "function":

        Yht = Yh.transpose()
        def process_batch(i):
            return tree.query(
                Yht[bins[i]:bins[i+1],:], 
                k=kk, 
                return_distance=True, 
                dualtree=dualtree, 
                breadth_first=breadth_first) 

        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(process_batch)(i)
            for i in range(n_batches))

    elif n_jobs_transfer_mode == "transfer_result_on_disk":

        import tempfile
        import pickle

        # manual pickle for large inter process communication load: 
        # https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
        Yht = Yh.transpose()
        with tempfile.TemporaryDirectory() as tmpdirname:

            # print("Temp dir:", tmpdirname)

            def process_bin(i):
                result = tree.query(
                    Yht[bins[i]:bins[i+1],:], 
                    k=kk, 
                    return_distance=True, 
                    dualtree=dualtree, 
                    breadth_first=breadth_first)
                pickle.dump(result, open(tmpdirname + "/result" + str(i), "wb"))

            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(process_bin)(i)
                for i in range(n_batches))

            results = [
                pickle.load(open(tmpdirname + "/result" + str(i), "rb"))
                for i in range(n_batches)
            ]

    elif n_jobs_transfer_mode == "transfer_all_on_disk":
        
        import tempfile
        import pickle

        # manual pickling for large inter process communication load: 
        # https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
        Yht = Yh.transpose()
        with tempfile.TemporaryDirectory() as tmpdirname:

            pickle.dump(tree, open(tmpdirname + "/tree", "wb"))
            for i in range(n_batches): 
                pickle.dump(Yht[bins[i]:bins[i+1],:], open(tmpdirname + "/bin" + str(i), "wb"))

            def process_bin(i):

                index = pickle.load(open(tmpdirname + "/tree", "rb"))
                query = pickle.load(open(tmpdirname + "/bin" + str(i), "rb"))

                result = index.query(
                    query, 
                    k=kk, 
                    return_distance=True, 
                    dualtree=dualtree, 
                    breadth_first=breadth_first)
                pickle.dump(result, open(tmpdirname + "/result" + str(i), "wb"))

            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(process_bin)(i)
                for i in range(n_batches))

            results = [
                pickle.load(open(tmpdirname + "/result" + str(i), "rb"))
                for i in range(n_batches)
            ]

    else:
        raise ValueError(f"Unknown transfer mode: {n_jobs_transfer_mode}")

    # TODO: 
    #   we could move the sorting into the parallel step 
    #   and then use a heap-based merge to combine these presorted arrays
    #   (see corals.sorting.sort.parallel_argsort)
    #   HOWEVER: this will require `query_sort=True`

    dst = np.concatenate([r[0] for r in results])
    idx = np.concatenate([r[1] for r in results])

    # normalize index
    idx_r = np.repeat(np.arange(len(idx)), [len(i) for i in idx])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Xh
    mask_inverse = idx_c >= Xh.shape[1]

    # fix index for correlations from -Xh
    idx_c[mask_inverse] -= Xh.shape[1]

    values, idx = derive_topk(
        mask_inverse, 
        np.concatenate(dst), 
        idx_r, 
        idx_c,
        k,
        threshold,
        argtopk_method=argtopk_method,
        require_sorted_topk=require_sorted_topk)
    # symmetrize to identify all topk values
    # TODO: it may make sense to only return the upper or lower triangle matrix in the first place; might be more efficient
    #       Actually I don't think that's possible. You will still have to sort the indices (see below)
    #       since the tree search returns values from the upper AND lower triangle.
    #       Thus, the symmetrization could be replaced by a "triangulization" which is similar to symmetrize
    #       but only returns the upper/lower triangle.
    if symmetrize:
        
        if Y is not None:
            raise ValueError("Symmetrization currently only supported for one set of features (as opposed to two).")
            # TODO: Implement this; outline: 
            #   1) run top-k search again (search(Y,X) in addition to search (X,Y))
            #   2) merge the results from both
            #   This obviously takes twice the amount of time. This may not be worth it in some cases.
            #   A top-k merging procedure would be really helpful for the merge.

        values, idx = symmetrize_topk(*sort_topk_idx(values, idx))

    return values, idx


def topk_balltree_combined_query(X, Y=None, 
        correlation_type="pearson", k=None, approximation_factor=10, tree_kwargs=None, dualtree=True, breadth_first=False):
    
    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, normalize=True, correlation_type=correlation_type, Y_fill_none=True)
    k = derive_k(Xh, Yh, k=k)
    kk = derive_k_per_query(Xh, Yh, k, approximation_factor)

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # build tree
    tree = sklearn.neighbors.BallTree(Xh.transpose(), **tree_kwargs)

    # run the query
    dst, idx = tree.query(np.concatenate([Yh, -Yh], axis=1).transpose(), k=kk, return_distance=True, dualtree=dualtree, breadth_first=breadth_first)

    # normalize index
    idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Yh
    mask_inverse = idx_r >= Yh.shape[1]

    # fix index for correlations from -Yh
    idx_r[mask_inverse] -= Yh.shape[1]

    return derive_topk(mask_inverse, np.concatenate(dst), idx_r, idx_c, k)


def topk_balltree_combined_query_parallel(
        X, Y=None, 
        correlation_type="pearson", 
        k=None, 
        approximation_factor=10, tree_kwargs=None, dualtree=True, breadth_first=False, 
        n_jobs=None, n_batches=None):
    """
    TODO: We can try distributed merge sort to speed up the merging process?
    """
    
    if n_jobs is None:
        n_jobs = 1

    if n_batches is None:
        n_batches = n_jobs

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, normalize=True, correlation_type=correlation_type, Y_fill_none=True)
    k = derive_k(Xh, Yh, k=k)
    kk = derive_k_per_query(Xh, Yh, k, approximation_factor)
    
    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # build tree
    tree = sklearn.neighbors.BallTree(Xh.transpose(), **tree_kwargs)

    # define query
    query = np.concatenate([Yh, -Yh], axis=1)

    # derive bins
    bins = derive_bins(query.shape[1], n_batches)

    # gather results
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(tree.query)(
            query[:, bins[i]:bins[i+1]].transpose(), 
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

    # get mask for selecting correlations from -Yh
    mask_inverse = idx_r >= Yh.shape[1]

    # fix index for correlations from -Yh
    idx_r[mask_inverse] -= Yh.shape[1]

    return derive_topk(mask_inverse, np.concatenate(dst), idx_r, idx_c, k)


def topk_balltree_twice(X, Y=None, 
        correlation_type="pearson", k=None, approximation_factor=10, tree_kwargs=None, dualtree=True, breadth_first=False):
    
    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, normalize=True, correlation_type=correlation_type, Y_fill_none=True)
    k = derive_k(Xh, Yh, k=k)
    kk = derive_k_per_query(Xh, Yh, k, approximation_factor)

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    tree = sklearn.neighbors.BallTree(Xh.transpose(), **tree_kwargs)

    # run the query
    dst1, idx1 = tree.query(Yh.transpose(), k=kk, return_distance=True, dualtree=dualtree, breadth_first=breadth_first)
    dst2, idx2 = tree.query(-Yh.transpose(), k=kk, return_distance=True, dualtree=dualtree, breadth_first=breadth_first)

    # normalize index
    idx1_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx1)])
    idx1_c = np.concatenate(idx1)

    idx2_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx2)])
    idx2_c = np.concatenate(idx2)

    idx_r = np.concatenate([idx1_r, idx2_r])
    idx_c = np.concatenate([idx1_c, idx2_c])

    mask_inverse = np.arange(len(idx1_r), len(idx1_r) + len(idx1_c))

    return derive_topk(mask_inverse, np.concatenate([*dst1, *dst2]), idx_r, idx_c, k)


def topk_balltree_positive(X, Y=None, 
        correlation_type="pearson", k=None, approximation_factor=10, tree_kwargs=None, dualtree=True, breadth_first=False):

    # set ball tree arguments
    if tree_kwargs is None:
        tree_kwargs = {}

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, normalize=True, correlation_type=correlation_type, Y_fill_none=True)
    k = derive_k(Xh, Yh, k=k)
    kk = derive_k_per_query(Xh, Yh, k, approximation_factor)

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    tree = sklearn.neighbors.BallTree(Xh.transpose(), **tree_kwargs)

    # run the query
    dst, idx1 = tree.query(Yh.transpose(), k=kk, return_distance=True, dualtree=dualtree, breadth_first=breadth_first)

    # normalize index
    idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx1)])
    idx_c = np.concatenate(idx1)

    # calculate correlations
    cor = 1 - np.concatenate(dst)**2 / 2

    # sort
    cor_order = np.argsort(-cor)

    # return    
    return cor[cor_order][:k], (idx_r[cor_order][:k], idx_c[cor_order][:k])

