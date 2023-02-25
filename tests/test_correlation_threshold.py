import numpy as np


test_top8_data = np.array([
    [1,1,-1,1],
    [2,1,-2,2],
    [3,3,-3,1],
    [4,4,-3,1],
])

test_top8_cor = [1,1,1,1,0.94672926,0.94672926,-0.94387981,-0.94387981]

test_top8_idx= [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 2),
    (3, 3)]


def test_fixed_cor_threshold_matrix_symmetrical():
    from corals.correlation.threshold._deprecated.original import cor_threshold_matrix_symmetrical
    cor1, _ = cor_threshold_matrix_symmetrical(test_top8_data, threshold=0.94)
    cor1 = cor1[np.argsort(-np.abs(cor1))]
    assert np.allclose(cor1, test_top8_cor)


def test_fixed_cor_threshold_balltree_combined_tree():
    from corals.correlation.threshold._deprecated.original import cor_threshold_balltree_combined_tree
    cor1, _ = cor_threshold_balltree_combined_tree(test_top8_data, threshold=0.94)
    cor1 = cor1[np.argsort(-np.abs(cor1))]
    assert np.allclose(cor1, test_top8_cor)


def test_fixed_cor_threshold_balltree_combined_query():
    from corals.correlation.threshold._deprecated.original import cor_threshold_balltree_combined_query
    cor1, _ = cor_threshold_balltree_combined_query(test_top8_data, threshold=0.94)
    cor1 = cor1[np.argsort(-np.abs(cor1))]
    assert np.allclose(cor1, test_top8_cor)


def test_fixed_cor_threshold_balltree_twice():
    from corals.correlation.threshold._deprecated.original import cor_threshold_balltree_twice
    cor1, _ = cor_threshold_balltree_twice(test_top8_data, threshold=0.94)
    cor1 = cor1[np.argsort(-np.abs(cor1))]
    assert np.allclose(cor1, test_top8_cor)


def test_cor_threshold_balltree_combined_tree():
    from corals.correlation.threshold._deprecated.original import cor_threshold_matrix_symmetrical, cor_threshold_balltree_combined_tree
    np.random.seed(5)
    X = np.random.random((10,4))
    X = np.concatenate([X, -X], axis=1)
    cor1, _ = cor_threshold_matrix_symmetrical(X, threshold=0.3)
    cor2, _ = cor_threshold_balltree_combined_tree(X, threshold=0.3)
    assert np.allclose(np.sort(cor1), np.sort(cor2))


def test_cor_threshold_balltree_combined_query():
    from corals.correlation.threshold._deprecated.original import cor_threshold_matrix_symmetrical, cor_threshold_balltree_combined_query
    np.random.seed(5)
    X = np.random.random((10,4))
    X = np.concatenate([X, -X], axis=1)
    cor1, _ = cor_threshold_matrix_symmetrical(X, threshold=0.3)
    cor2, _ = cor_threshold_balltree_combined_query(X, threshold=0.3)
    assert np.allclose(np.sort(cor1), np.sort(cor2))


def test_cor_threshold_balltree_twice():
    from corals.correlation.threshold._deprecated.original import cor_threshold_matrix_symmetrical, cor_threshold_balltree_twice
    np.random.seed(5)
    X = np.random.random((10,4))
    X = np.concatenate([X, -X], axis=1)
    cor1, _ = cor_threshold_matrix_symmetrical(X, threshold=0.3)
    cor2, _ = cor_threshold_balltree_twice(X, threshold=0.3)
    assert np.allclose(np.sort(cor1), np.sort(cor2))


def test_fixed_cor_threshold_balltree_combined_query_parallel():
    from corals.correlation.threshold._deprecated.original import cor_threshold_balltree_combined_query_parallel
    cor1, _ = cor_threshold_balltree_combined_query_parallel(
        test_top8_data, threshold=0.94, n_jobs=4)
    cor1 = cor1[np.argsort(-np.abs(cor1))]
    assert np.allclose(cor1, test_top8_cor)
