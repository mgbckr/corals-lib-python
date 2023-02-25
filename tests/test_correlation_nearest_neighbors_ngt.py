import pytest


@pytest.mark.experimental
def test_ngt():

    import timeit
    import numpy as np
    import sklearn.neighbors

    from corals.correlation.utils import preprocess_XY
    from corals.correlation.topk.batched.nearest_neighbors_ann_ngt import Ngt
    
    def test(X, n_runs=100, k=5):
        print(f"Experiment: {X.shape}")

        bt = sklearn.neighbors.BallTree(X)
        bt_result = bt.query(X, k=k)

        ann = Ngt(X)
        ann_result = ann.query(X, k=k)

        assert np.all(np.isclose(ann_result[0], bt_result[0]))
        assert np.all(np.isclose(ann_result[1], bt_result[1]))

    test(np.random.random((100, 100)))

    test_top8_data = np.array([
        [1,1,-1,1],
        [2,1,-2,2],
        [3,3,-3,1],
        [4,4,-3,1],
    ])
    X, Y = preprocess_XY(test_top8_data.transpose())
    test(X, k=4)


if __name__ == '__main__':
    test = 'test_ngt'
    if test in globals():
        globals()[test]()
        pass
