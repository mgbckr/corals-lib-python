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


def test_topk_general():

    import functools
    import tempfile
    from corals.correlation.topk.baselines import topk_full_corrcoef, topk_indexed_loop
    # from corals.correlation.topk._experimental.distributed import topk_dask_delayed
    # from corals.correlation.topk._experimental.nndescent import topk_nndescent_direct

    with tempfile.TemporaryDirectory() as temp_dir:

        topk_functions = [
            ("corccoef", topk_full_corrcoef),
            ("indexed_loop", topk_indexed_loop),
            # ("task_delayed", topk_dask_delayed),
            # ("nndescent_direct", topk_nndescent_direct),  # TODO: currently does not work
        ]

        for name, topk in topk_functions:
            print(name)
            val, idx = topk(test_top8_data, k=8)
            assert np.allclose(val, test_top8_cor)
            assert sorted(zip(*idx)) == test_top8_idx


if __name__ == '__main__':
    test = 'test_topk_general'
    if test in globals():
        globals()[test]()
        pass


