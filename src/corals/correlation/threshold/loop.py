import numba
import numpy as np
from corals.correlation.utils import preprocess_XY


@numba.jit(nopython=True)
def _threshold_indexed_loop_calculate(Xh, Yh, threshold=None):
    """ 
        TODO: would be nice to parallelize
        However, parallel looping (`numba.jit(parall=True)`) using `numba.prange` does not seem to be an option.
        This is because `list.append` does not work in `numba.prange`.
        As a result chunked matrix multiplication might actually be the best option for implementing this in pure Python.
    """

    results = []
    for i in range(Xh.shape[0]):
        for j in range(i + 1, Yh.shape[0]):
            r = Xh[i,:] @ Yh[j,:]
            if threshold is None or abs(r) >= threshold:
                results.append((i, j, r))

    correlations = np.array([r for _, _, r in results])
    row_idx = np.array([i for i, _, _ in results])
    col_idx = np.array([j for _, j, _ in results])

    return correlations, (row_idx, col_idx)


def threshold_indexed_loop(X, Y=None, correlation_type="pearson", threshold=None, mode="all", **kwargs):
    """ TODO: Since no parallelization is involved, this is pretty slow. """

    # preprocess matrices
    Xh, Yh = preprocess_XY(X, Y, correlation_type=correlation_type, normalize=True, Y_fill_none=True)

    print("calculate correlation results")
    correlations, (row_idx, col_idx) = _threshold_indexed_loop_calculate(
        Xh.transpose().copy(), Yh.transpose().copy(), threshold=threshold)

    # add lower matrix if requested
    if mode == "all" or mode == "mirror":
        correlations = np.concatenate([correlations, correlations])
        new_row_idx = np.concatenate([row_idx, col_idx])
        col_idx = np.concatenate([col_idx, row_idx])
        row_idx = new_row_idx

    if mode == "all":
        n_diag = min(Xh.shape[1], Yh.shape[1])
        correlations = np.concatenate([correlations, np.ones(n_diag)])
        row_idx = np.concatenate([row_idx, np.arange(n_diag)])
        col_idx = np.concatenate([col_idx, np.arange(n_diag)])

    return correlations, (row_idx, col_idx)
