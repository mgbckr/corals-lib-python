import numpy as np


def test_sort_idx():

    import corals.correlation.utils_numba

    # sorting
    v =     np.array([1,1,1,1,4])
    idx_r = np.array([1,3,9,4,4])
    idx_c = np.array([9,4,1,3,4])

    v2 =     np.array([1,1,1,1,4])
    idx_r2 = np.array([1,9,3,4,4])
    idx_c2 = np.array([9,1,4,3,4])

    v3, (idx_r3, idx_c3) = corals.correlation.utils_numba.sort_topk_idx(v, (idx_r, idx_c))

    print(v3)
    print(idx_r3)
    print(idx_c3)

    assert np.allclose(v2, v3)
    assert np.allclose(idx_r2, idx_r3)
    assert np.allclose(idx_c2, idx_c3)


def test_symmetrize(): 

    from corals.correlation.utils_numba import symmetrize_topk, sort_topk_idx 

    v =     np.array([1,1,3,4,5])
    idx_r = np.array([1,2,3,4,5])
    idx_c = np.array([2,1,3,4,5])

    v2 =     np.array([1,1,3,4,5])
    idx_r2 = np.array([1,2,3,4,5])
    idx_c2 = np.array([2,1,3,4,5])
    
    v3, (idx_r3, idx_c3) = symmetrize_topk(v, (idx_r, idx_c))
    
    assert np.allclose(v2, v3)
    assert np.allclose(idx_r2, idx_r3)
    assert np.allclose(idx_c2, idx_c3)
    
    v =     np.array([1,1,3,4,5])
    idx_r = np.array([1,2,3,4,5])
    idx_c = np.array([2,1,4,4,5])

    v2 =     np.array([1,1,3,3,4])
    idx_r2 = np.array([1,2,3,4,4])
    idx_c2 = np.array([2,1,4,3,4])
    
    v3, (idx_r3, idx_c3) = symmetrize_topk(v, (idx_r, idx_c))
    
    assert np.allclose(v2, v3)
    assert np.allclose(idx_r2, idx_r3)
    assert np.allclose(idx_c2, idx_c3)
    
    v =     np.array([1,1,3])
    idx_r = np.array([1,2,3])
    idx_c = np.array([2,1,4])

    v2 =     np.array([1,1,3])
    idx_r2 = np.array([1,2,3])
    idx_c2 = np.array([2,1,4])
    
    v3, (idx_r3, idx_c3) = symmetrize_topk(v, (idx_r, idx_c))
    
    assert np.allclose(v2, v3)
    assert np.allclose(idx_r2, idx_r3)
    assert np.allclose(idx_c2, idx_c3)
    
    # sorting
    v =     np.array([1,1,1,1,5])
    idx_r = np.array([1,3,9,4,5])
    idx_c = np.array([9,4,1,3,5])

    v2 =     np.array([1,1,1,1,5])
    idx_r2 = np.array([1,9,3,4,5])
    idx_c2 = np.array([9,1,4,3,5])
    
    print(sort_topk_idx(v, (idx_r, idx_c)))
    v3, (idx_r3, idx_c3) = symmetrize_topk(*sort_topk_idx(v, (idx_r, idx_c)))
    print(v3, (idx_r3, idx_c3))
    
    assert np.allclose(v2, v3)
    assert np.allclose(idx_r2, idx_r3)
    assert np.allclose(idx_c2, idx_c3)
    