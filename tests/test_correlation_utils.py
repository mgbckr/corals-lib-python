import numpy as np

from corals.correlation.utils import derive_bins


def test_parallel_argsort():

    from corals.sorting.parallel import parallel_argsort

    np.random.seed(142)
    a = np.random.random(1000)

    order = parallel_argsort(a)
    assert np.allclose(order, np.argsort(a))

    order = parallel_argsort(a, n_jobs=4)
    assert np.allclose(order, np.argsort(a))

    order = parallel_argsort(a, n_jobs=4, n_batches=10)
    assert np.allclose(order, np.argsort(a))


def test_topksort():

    from corals.correlation.utils import argtopk

    np.random.seed(142)
    a = np.random.random(1000)

    a_order = np.argsort(a)
    a_sorted = a[a_order]

    idx, values = argtopk(a, k=10, argtopk_method="argpartition", return_values=True)
    assert np.array_equiv(a_sorted[:10], values)
    assert np.array_equiv(a_order[:10], idx)

    idx, values = argtopk(a, k=10, argtopk_method="argsort", return_values=True)
    assert np.array_equiv(a_sorted[:10], values)
    assert np.array_equiv(a_order[:10], idx)

    idx, values = argtopk(a, k=10, threshold=0.9, argtopk_method="argpartition", return_values=True)
    assert np.array_equiv(a_sorted[:10], values)
    assert np.array_equiv(a_order[:10], idx)

    # test for same size array as topk

    np.random.seed(142)
    a = np.random.random(10)

    a_order = np.argsort(a)
    a_sorted = a[a_order]

    idx, values = argtopk(a, k=10, argtopk_method="argpartition", return_values=True)
    assert np.array_equiv(a_sorted[:10], values)
    assert np.array_equiv(a_order[:10], idx)

    idx, values = argtopk(a, k=10, argtopk_method="argsort", return_values=True)
    assert np.array_equiv(a_sorted[:10], values)
    assert np.array_equiv(a_order[:10], idx)

    # test for a single value

    idx, values = argtopk(a, k=1, argtopk_method="argpartition", return_values=True)
    assert np.array_equiv(a_sorted[:1], values)
    assert np.array_equiv(a_order[:1], idx)

    idx, values = argtopk(a, k=1, argtopk_method="argsort", return_values=True)
    assert np.array_equiv(a_sorted[:1], values)
    assert np.array_equiv(a_order[:1], idx)


def test_pvalues():
    
    from corals.correlation.utils import derive_pvalues
    import scipy.stats
    
    X = np.random.random((10, 5))
    correlations = []
    pvalues = []
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            r, p = scipy.stats.pearsonr(X[:,i], X[:,j])
            correlations.append(r)
            pvalues.append(p)
    correlations = np.asarray(correlations)
    pvalues = np.asarray(pvalues)

    derived_pvalues = derive_pvalues(correlations, X.shape[0])
    assert np.all(np.isclose(pvalues, derived_pvalues))

def test_multiple_test_correction():
    
    from corals.correlation.utils import multiple_test_correction
    import numpy as np
    import statsmodels.stats.multitest
    
    rnd = np.random.RandomState(42)
    pvalues = rnd.random(4 * 4) / 10
    
    _, pvalues_bonferroni, _, _ = statsmodels.stats.multitest.multipletests(pvalues, method="bonferroni")
    _, pvalues_fdr_bh, _, _ = statsmodels.stats.multitest.multipletests(pvalues, method="fdr_bh")
    
    pvalues_bonferroni_test = multiple_test_correction(pvalues.flatten(), 4, "bonferroni", minimal_pvalues=False)
    pvalues_fdr_bh_test = multiple_test_correction(pvalues.flatten(), 4, "fdr_bh", minimal_pvalues=False)
    
    # print(pvalues)
    # print(pvalues_fdr_bh)
    # print(pvalues_fdr_bh_test)
    
    assert np.array_equiv(pvalues_bonferroni, pvalues_bonferroni_test)
    assert np.array_equiv(pvalues_fdr_bh, pvalues_fdr_bh_test)
    
if __name__ == '__main__':
    test = 'test_multiple_test_correction'
    if test in globals():
        globals()[test]()
        pass
