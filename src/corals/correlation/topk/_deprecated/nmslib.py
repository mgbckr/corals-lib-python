import numpy as np
import nmslib

from corals.correlation.utils import preprocess_XY, derive_k, derive_k_per_query, derive_topk


def topk_balltree_combined_tree_nmslib(X, Y=None, k=None, approximation_factor=10, n_jobs=1):


    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y)
    k = derive_k(Xh, Yh, k=k)
    kk = derive_k_per_query(Xh, Yh, k, approximation_factor)

    # build tree; note that we concatenate Xh and -Xh to capture negative correlations
    index = nmslib.init(method='hnsw', space='l2',)
    index.addDataPointBatch(np.concatenate([Xh, -Xh], axis=1).transpose())
    index.createIndex({'post': 2, "indexThreadQty": n_jobs}, print_progress=True)

    # run query    
    neighbors = index.knnQueryBatch(Yh.transpose(), k=kk, num_threads=n_jobs)
    idx = [i for i, _ in neighbors]
    dst = [d for _, d in neighbors]

    # normalize index
    idx_r = np.concatenate([np.repeat(i, len(indices)) for i, indices in enumerate(idx)])
    idx_c = np.concatenate(idx)

    # get mask for selecting correlations from -Xh
    mask_inverse = idx_c >= Xh.shape[1]

    # fix index for correlations from -Xh
    idx_c[mask_inverse] -= Xh.shape[1]

    return derive_topk(mask_inverse, np.concatenate(dst), idx_r, idx_c,k)
    