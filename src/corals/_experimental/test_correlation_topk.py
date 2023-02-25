import corals.threads
corals.threads.set_threads_for_external_libraries(1)


from re import S
import numpy as np

from dask.distributed import Client

client = Client(processes=False)
# client = Client("10.110.6.47:8786")
import joblib
# backend = "loky"
# backend = "threading"
backend = "dask"

test_top8_data = np.array([
    [1,1,-1,1],
    [2,1,-2,2],
    [3,3,-3,1],
    [4,4,-3,1],
])
test_top8_data = np.random.random((68,64000))

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


def test_topk_generic():

    import functools
    import tempfile

    from corals.correlation.topk.distributed import topk_dask, topk_dask_simple, topk_dask_delayed

    from corals.correlation.topk._deprecated.original import topk_balltree_combined_tree_parallel_optimized

    from corals.correlation.topk.batched.base import \
        topk_batched_generic
    # from corals.correlation.topk.batched.nearest_neighbors_ann_ngt import \
    #     NgtTopkMapReduce
    # from corals.correlation.topk.batched.nearest_neighbors_ann_nmslib import \
    #     NmslibTopkMapReduce
    from corals.correlation.topk.batched.matmul import \
        MatmulTopkMapReduce
    from corals.correlation.topk.batched.nearest_neighbors_balltree import \
        BalltreeTopkMapReduce

    with tempfile.TemporaryDirectory() as temp_dir:

        topk_functions = [
            # ("balltree", functools.partial(
            #     topk_batch_generic, 
            #     threshold=0.7,
            #     k=8, 
            #     #
            #     approximation_factor=10,
            #     n_batches=2,
            #     n_jobs=1,
            #     #
            #     # mapreduce=BalltreeQueryResultTopkMapReduce()
            #     mapreduce=BalltreeTopkMapReduce()
            #     # batch_topk_method=corals.correlation.topk_split_generic.batch_topk_pickle,
            #     # batch_topk_method_kwargs=dict(
            #     #     out_folder="/tmp"
            #     # ),
            #     # mapreduce=MatmulTopkMapReduce(
            #     #     # reducer=
            #     # )
            #     # batch_topk_method=batch_map_matmul,
            #     # batch_topk_method_kwargs=dict(
            #     #     # out_folder="/tmp"
            #     # ),
            #     # #
            #     # batch_reduce_method=batch_reduce_iterative_concatenate,
            #     # batch_reduce_method_kwargs=dict(
            #     #     n_chunks=2,
            #     #     # load_result=load_result_pickle,
            #     #     # cleanup_result=cleanup_result_delete
            #     # )
            # )),
            # ("balltree_test_original", functools.partial(
            #     topk_balltree_combined_tree_parallel_optimized, 
            #     k=0.001, 
            #     #
            #     approximation_factor=10,
            #     n_batches=64,
            #     n_jobs=64,
            #     n_jobs_transfer_mode="direct"
            # )),
            # ("balltree_test_new", functools.partial(
            #     topk_batched_generic, 
            #     threshold=None,
            #     k=0.001, 
            #     #
            #     approximation_factor=10,
            #     n_batches=64,
            #     n_jobs=64,
            #     #
            #     # mapreduce=BalltreeJobFocusedTopkMapReduce(),
            #     mapreduce=BalltreeTopkMapReduce(),
            #     # mapreduce=MatmulTopkMapReduce(),
            # )),
            # ("nmslib_test", functools.partial(
            #     topk_batched_generic, 
            #     threshold=None,
            #     k=0.001, 
            #     #
            #     approximation_factor=10,
            #     n_batches=64,
            #     n_jobs=64,
            #     #
            #     # mapreduce=BalltreeJobFocusedTopkMapReduce(),
            #     mapreduce=NmslibTopkMapReduce(),
            #     # mapreduce=MatmulTopkMapReduce(),
            # )),
            # ("ngt_test", functools.partial(
            #     topk_batched_generic, 
            #     threshold=None,
            #     k=0.001, 
            #     #
            #     approximation_factor=10,
            #     n_batches=64,
            #     n_jobs=64,
            #     #
            #     mapreduce=NgtTopkMapReduce(
            #         n_threads_build_index=64
            #     ),
            #     # preferred_backend="threads"
            # )),
            # ("ngt_test", functools.partial(
            #     topk_batched_generic, 
            #     threshold=None,
            #     k=8, 
            #     #
            #     approximation_factor=None,
            #     n_batches=1,
            #     n_jobs=1,
            #     #
            #     # mapreduce=BalltreeTopkMapReduce()
            #     mapreduce=NgtTopkMapReduce(
            #         n_threads_build_index=1
            #     ),
            #     # preferred_backend="threads"
            # )),
            # ("dask", functools.partial(
            #     topk_dask_simple, 
            #     threshold=0.7,
            #     k=0.001,
            #     #
            #     approximation_factor=None,
            #     n_batches=1000,
            # )),
            # ("dask_simple", functools.partial(
            #     topk_dask_simple, 
            #     threshold=0.9,
            #     k=0.001,
            #     #
            #     chunks=1000,
            # )),
            # ("dask_delayed", functools.partial(
            #     topk_dask_delayed, 
            #     threshold=None,
            #     k=0.001,
            # )),
            # ("balltree_reduce_load", functools.partial(
            #     topk_batch_generic,
            #     threshold=0.7,
            #     k=8, 
            #     #
            #     approximation_factor=10,
            #     n_batches=2,
            #     n_jobs=1,
            #     #
            #     mapreduce=BalltreeReduceLoadTopkMapReduce()
            # )),
            ("matmul", functools.partial(
                topk_batched_generic,
                threshold=0.7,
                k=0.01, 
                #
                approximation_factor=None,
                n_batches=5000,
                n_jobs=64,
                # attempt_limiting_blas_threads_in_map=1,
                #
                mapreduce=MatmulTopkMapReduce()
            ))
        ]

        for name, topk in topk_functions:
            print(name)
            from threadpoolctl import threadpool_info
            print(threadpool_info())
            with joblib.parallel_backend(backend):

                import timeit
                t = timeit.timeit(lambda: topk(test_top8_data), number=1)
                print(t)

            # val, idx = topk(test_top8_data)
            # assert sorted(zip(*idx)) == test_top8_idx
            # assert np.allclose(val, test_top8_cor)


if __name__ == '__main__':
    test = 'test_topk_generic'
    if test in globals():
        globals()[test]()
        pass


