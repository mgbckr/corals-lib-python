import numpy as np


def test_topdiff_find_all():
    """Test whether the approach works when returning ALL possible differences"""

    for correlation_type in ["pearson", "spearman"]:
        for i in range(100):

            import corals.correlation.topkdiff.original
            import corals.correlation.full.matmul

            # np.random.seed(3)
            np.random.seed(i)
            print(i)

            m,n = 20, 10

            m1 = (np.random.random((m,n)) - 0.5) * 10
            m2 = (np.random.random((m,n)) - 0.5) * 10
            
            cm1 = corals.correlation.full.matmul.full_matmul_symmetrical(m1 ,correlation_type=correlation_type)
            cm2 = corals.correlation.full.matmul.full_matmul_symmetrical(m2 ,correlation_type=correlation_type)
            

            m_diff = cm1 - cm2
            v_diff = m_diff.flatten()
            
            order = np.argsort(-np.abs(v_diff))

            topkdiff, _ = corals.correlation.topkdiff.original.topkdiff_balltree_combined_tree_parallel(m1, m2, k=n**2, correlation_type=correlation_type)

            assert np.allclose(np.abs(v_diff[order]), np.abs(topkdiff))


def test_topdiff_find_column():
    """Test whether for one query column our the approach works (not approximate)"""

    for correlation_type in ["pearson", "spearman"]:
        for i in range(10):

            import corals.correlation.topkdiff.original
            import corals.correlation.full.matmul

            # np.random.seed(3)
            np.random.seed(i)
            print(i)

            m,n = 20, 100

            x1 = (np.random.random((m,n)) - 0.5) * 10
            y1 = (np.random.random((m,1)) - 0.5) * 10

            x2 = (np.random.random((m,n)) - 0.5) * 10
            y2 = (np.random.random((m,1)) - 0.5) * 10
            
            m1 = (x1, y1)
            m2 = (x2, y2)

            cm1 = corals.correlation.full.matmul.full_matmul_symmetrical(x1, y1 ,correlation_type=correlation_type)
            cm2 = corals.correlation.full.matmul.full_matmul_symmetrical(x2, y2 ,correlation_type=correlation_type)

            m_diff = cm1 - cm2
            v_diff = m_diff.flatten()
            
            order = np.argsort(-np.abs(v_diff))

            k = 20
            topkdiff, _ = corals.correlation.topkdiff.original.topkdiff_balltree_combined_tree_parallel(m1, m2, k=k, correlation_type=correlation_type)

            assert np.allclose(np.abs(v_diff[order])[:k], np.abs(topkdiff))
    
if __name__ == '__main__':
    test = 'test_topdiff_find_column'
    if test in globals():
        globals()[test]()
        pass
