def test_loop():

    import numpy as np
    from corals.correlation.threshold.loop import threshold_indexed_loop

    np.random.seed(42)

    n, m = 256, 1000
    X = np.random.random((n, m))

    ref_cor = np.corrcoef(X, rowvar=False)
    ref_cor_idx = np.triu_indices_from(ref_cor, 1)
    ref_cor_flat = ref_cor[ref_cor_idx].flatten()
    ref_cor_order = np.argsort(ref_cor_flat)    
    
    cor_flat, (row_idx, col_idx) = threshold_indexed_loop(X, mode=None)
    cor_order = np.argsort(cor_flat)

    assert cor_flat[cor_order].size == ref_cor_flat[ref_cor_order].size
    assert np.all(np.isclose(cor_flat[cor_order], ref_cor_flat[ref_cor_order]))

    threshold = 0.5

    ref_cor_order = ref_cor_order[np.abs(ref_cor_flat[ref_cor_order]) > threshold]
    
    cor_flat, (row_idx, col_idx) = threshold_indexed_loop(X, mode=None, threshold=threshold)
    cor_order = np.argsort(cor_flat)

    assert cor_flat[cor_order].size == ref_cor_flat[ref_cor_order].size
    assert np.all(np.isclose(cor_flat[cor_order], ref_cor_flat[ref_cor_order]))
    


if __name__ == '__main__':
    test = 'test_loop'
    if test in globals():
        globals()[test]()
        pass
