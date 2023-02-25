import numpy as np
from corals.correlation.utils import preprocess_XY


def full_corrcoef(X, Y=None, correlation_type="pearson", **kwargs):
    Xh, _ = preprocess_XY(X, correlation_type=correlation_type, normalize=False, Y_fill_none=False)
    return np.corrcoef(Xh, rowvar=False)
