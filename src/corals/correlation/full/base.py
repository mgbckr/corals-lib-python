def full(X, Y=None, correlation_type="pearson", **kwargs):
    raise NotImplementedError()


# convenience import for end user
from .matmul import full_matmul_asymmetrical as cor_matrix