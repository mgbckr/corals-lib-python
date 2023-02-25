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

test_top8_idx_threshold= [
    (0, 0),
    (0, 1),
    # (0, 2),
    (1, 0),
    (1, 1),
    # (2, 0),
    (2, 2),
    (3, 3)]


def test_topk_corrcoef():
    from corals.correlation.topk._deprecated.original import topk_corrcoef
    val, idx = topk_corrcoef(test_top8_data, k=8)
    assert np.allclose(val, test_top8_cor)
    assert sorted(zip(*idx)) == test_top8_idx


def test_topk_matrix():
    from corals.correlation.topk._deprecated.original import topk_matrix
    val, idx = topk_matrix(test_top8_data, k=8)
    assert np.allclose(val, test_top8_cor)
    assert sorted(zip(*idx)) == test_top8_idx


def test_topk_balltree_combined_tree():
    from corals.correlation.topk._deprecated.original import topk_balltree_combined_tree
    val, idx = topk_balltree_combined_tree(test_top8_data, k=8)
    assert np.allclose(val, test_top8_cor)
    assert sorted(zip(*idx)) == test_top8_idx


def test_topk_balltree_combined_tree_parallel():
    from corals.correlation.topk._deprecated.original import topk_balltree_combined_tree_parallel
    val, idx = topk_balltree_combined_tree_parallel(test_top8_data, k=8)
    assert np.allclose(val, test_top8_cor)
    assert sorted(zip(*idx)) == test_top8_idx
    
    
def test_topk_balltree_combined_tree_parallel_optimized___regular():
    from corals.correlation.topk._deprecated.original import topk_balltree_combined_tree_parallel_optimized
    val, idx = topk_balltree_combined_tree_parallel_optimized(test_top8_data, k=8)
    assert np.allclose(val, test_top8_cor)
    assert sorted(zip(*idx)) == test_top8_idx
    
    
def test_topk_balltree_combined_tree_parallel_optimized___threshold():
    from corals.correlation.topk._deprecated.original import topk_balltree_combined_tree_parallel_optimized
    val, idx = topk_balltree_combined_tree_parallel_optimized(test_top8_data, k=8, threshold=0.946)
    assert np.allclose(val, test_top8_cor[:-2])
    assert sorted(zip(*idx)) == test_top8_idx_threshold
    

def test_topk_balltree_combined_query():
    from corals.correlation.topk._deprecated.original import topk_balltree_combined_query
    val, idx = topk_balltree_combined_query(test_top8_data, k=8)
    assert np.allclose(val, test_top8_cor)
    assert sorted(zip(*idx)) == test_top8_idx


def test_topk_balltree_twice():
    from corals.correlation.topk._deprecated.original import topk_balltree_twice
    val, idx = topk_balltree_twice(test_top8_data, k=8)
    assert np.allclose(val, test_top8_cor)
    assert sorted(zip(*idx)) == test_top8_idx


def test_topk_balltree_combined_query_parallel():
    from corals.correlation.topk._deprecated.original import topk_balltree_combined_query_parallel
    val, idx = topk_balltree_combined_query_parallel(test_top8_data, k=8, n_jobs=4)
    assert np.allclose(val, test_top8_cor)
    assert sorted(zip(*idx)) == test_top8_idx


def test_topk_matrix_parallel_sort():
    from corals.correlation.topk._deprecated.original import topk_matrix_parallel_sort
    val, idx = topk_matrix_parallel_sort(test_top8_data, k=8, n_jobs=2)
    assert np.allclose(val, test_top8_cor)
    assert sorted(zip(*idx)) == test_top8_idx

