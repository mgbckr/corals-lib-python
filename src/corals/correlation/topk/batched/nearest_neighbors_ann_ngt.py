import ngtpy
import shutil
import numpy as np

from corals.correlation.topk.batched.base import TopkMapReduce, ResultHandler
from corals.correlation.topk.batched.nearest_neighbors import NearestNeighborTopkMapReduce, NearestNeighborJobFocusedTopkMapReduce


class Ngt():
    """
    ONNG seems to be the fastest NGT variant (see: https://github.com/erikbern/ann-benchmarks), 
    however it is not available yet via the Python API (see: https://github.com/yahoojapan/NGT/blob/master/python/README-ngtpy.md).
    """

    def __init__(self, X=None, index_path=None, n_threads=8):
        """
        TODO: allow parameter passing and manage index folder better (particularly deleting it after use?)
        """

        index_path = "tmp_index"

        if X is not None:
            shutil.rmtree(index_path, ignore_errors=True)
            ngtpy.create(index_path, dimension=X.shape[1])
            index = ngtpy.Index(index_path)
            # NOTE: we have to `.copy()` X here since for some reason if you pass a view (e.g., from X.transpose()) things break ... interesting ...
            index.batch_insert(X.copy(), num_threads=n_threads)
            index.save()
        else: 
            index = ngtpy.Index(index_path)
        
        self.index_ = index

    def query(self, queries, k=1, return_distance=True):

        results = [self.index_.search(q, size=k) for q in queries]
        distances = np.array([[d for _, d in r] for r in results])
        idx = np.array([[i for i, _ in r] for r in results])

        if return_distance:
            return distances, idx
        else:
            return idx


class NgtTopkMapReduce(NearestNeighborTopkMapReduce):

    def __init__(
            self, 
            n_threads_build_index=8,
            argtopk_method="argpartition",
            result_handler: ResultHandler=None) -> None:

        super().__init__(
            None, None, dict(),
            argtopk_method=argtopk_method,
            result_handler=result_handler)
        self.n_threads_build_index = n_threads_build_index

    def prepare_index(self, X, Y, threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        Ngt(X=np.concatenate([X, -X], axis=1).transpose(), index_path="tmp_index", n_threads=self.n_threads_build_index)

    def load_index(self):
        return Ngt(index_path="tmp_index")


class NgtJobFocusedTopkMapReduce(NearestNeighborJobFocusedTopkMapReduce):

    def __init__(
            self, 
            n_threads_build_index=8,
            result_handler: ResultHandler=None,
            reducer: TopkMapReduce = None) -> None:

        super().__init__(
            None, None, dict(),
            result_handler=result_handler,
            reducer=reducer)
        self.n_threads_build_index = n_threads_build_index
    
    def prepare_index(self, X, Y, threshold, k, n_batches, approximation_factor, result_handler: ResultHandler):
        Ngt(np.concatenate([X, -X], axis=1).transpose(), index_path="tmp_index", n_threads=self.n_threads_build_index)

    def load_index(self):
        return Ngt(index_path="tmp_index", overwrite=False)


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

        ann = Ngt(X)
        ann_result = ann.query(X[:20], k=5)
        ann_t = timeit.timeit(lambda: ann.query(X[:20], k=5), number=n_runs)
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

