import nmslib
import numpy as np

from corals.correlation.topk.batched.base import TopkMapReduce, ResultHandler
from corals.correlation.topk.batched.nearest_neighbors import NearestNeighborTopkMapReduce, NearestNeighborJobFocusedTopkMapReduce


class Nmslib():
    """
    """

    def __init__(self, X=None, index=None):
        """
        TODO: allow parameter passing and manage index folder better (particularly deleting it after use?)
        Source: https://github.com/nmslib/nmslib/blob/master/python_bindings/notebooks/search_vector_dense_optim.ipynb
        """

        if index is None:

            # Set index parameters
            # These are the most important onese
            M = 15
            efC = 100

            num_threads = 4
            index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 2}
            # print('Index-time parameters', index_time_params)

            # Space name should correspond to the space name 
            # used for brute-force search
            space_name='l2'

            # Intitialize the library, specify the space, the type of the vector and add data points 
            index = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
            index.addDataPointBatch(X)

            # Create an index
            # start = time.time()
            index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC}
            index.createIndex(index_time_params) 
            # end = time.time()
            # print('Index-time parameters', index_time_params)
            # print('Indexing time = %f' % (end-start))

            # Setting query-time parameters
            efS = 100
            query_time_params = {'efSearch': efS}
            # print('Setting query-time parameters', query_time_params)
            index.setQueryTimeParams(query_time_params)

            # # Save a meta index, but no data!
            # index.saveIndex('dense_index_optim.bin', save_data=False)

            # # Re-intitialize the library, specify the space, the type of the vector.
            # newIndex = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR) 
            # # For an optimized L2 index, there's no need to re-load data points, but this would be required for
            # # non-optimized index or any other methods different from HNSW (other methods can save only meta indices)
            # #newIndex.addDataPointBatch(data_matrix) 

            # # Re-load the index and re-run queries
            # newIndex.loadIndex('dense_index_optim.bin')

            # Setting query-time parameters and querying
            # print('Setting query-time parameters', query_time_params)
            # newIndex.setQueryTimeParams(query_time_params)

            # self.index_ = newIndex
        
        self.index_ = index

    def query(self, queries, k=1, return_distance=True, n_threads=1):

        results = self.index_.knnQueryBatch(queries, k=k, num_threads=n_threads)

        distances = np.sqrt(np.concatenate([np.pad(d, (0, k - d.size), constant_values=np.nan).reshape(1, -1) for _, d in results]))
        idx = np.concatenate([np.pad(i, (0, k - i.size), constant_values=-1).reshape(1, -1) for i, _ in results])

        if return_distance:
            return distances, idx
        else:
            return idx


class NmslibTopkMapReduce(NearestNeighborTopkMapReduce):

    def __init__(
            self, 
            argtopk_method="argpartition",
            result_handler: ResultHandler=None) -> None:

        super().__init__(
            None, None, dict(),
            argtopk_method=argtopk_method,
            result_handler=result_handler)

    def prepare_index(self, X, Y, threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        # self.index_ = Nmslib(X=np.concatenate([X, -X], axis=1).transpose())
        index = Nmslib(X=np.concatenate([X, -X], axis=1).transpose())        
        index.index_.saveIndex('dense_index_optim.bin', save_data=True)

    def load_index(self):
        index = nmslib.init(method='hnsw', space="l2", data_type=nmslib.DataType.DENSE_VECTOR)
        index.loadIndex('dense_index_optim.bin')
        return Nmslib(index=index)


class NmslibJobFocusedTopkMapReduce(NearestNeighborJobFocusedTopkMapReduce):

    def __init__(
            self, 
            result_handler: ResultHandler=None,
            reducer: TopkMapReduce = None) -> None:

        super().__init__(
            None, None, dict(),
            result_handler=result_handler,
            reducer=reducer)
    
    def prepare_index(self, X, Y, threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        self.index_ = Nmslib(X=np.concatenate([X, -X], axis=1).transpose().copy())

    def load_index(self):
        return self.index_


if __name__ == '__main__':

    import timeit
    import numpy as np
    import sklearn.neighbors
    
    def test(n_samples, n_features, n_runs=100):
        print(f"Experiment: {n_samples}, {n_features}")

        X = np.random.random((n_samples, n_features))

        bt = sklearn.neighbors.BallTree(X)
        bt_result = bt.query(X[:20], k=5)
        bt_t = timeit.timeit(lambda: bt.query(X[:20], k=5), number=n_runs)
        print("* balltree:", bt_t)

        ann = Nmslib(X)
        ann_result = ann.query(X[:20], k=5)
        ann_t = timeit.timeit(lambda: ann.query(X[:20].copy(), k=5), number=n_runs)
        print("* NGT:     ", ann_t)

    test(n_samples=10, n_features=100)
    test(n_samples=100, n_features=100)
    test(n_samples=1000, n_features=100)
    test(n_samples=10000, n_features=100)  # this is actually interesting for our usecase (but only for large batches), but we would need to figure out accuracy!

    test(n_samples=100, n_features=10)
    test(n_samples=100, n_features=100)
    test(n_samples=100, n_features=1000)
    test(n_samples=100, n_features=10000)

    test(n_samples=10000, n_features=10)
    test(n_samples=10000, n_features=100)
    test(n_samples=10000, n_features=1000, n_runs=10)  # here NGT is actually slower!!!
    # test(n_samples=10000, n_features=10000)  # takes too long


    pass

