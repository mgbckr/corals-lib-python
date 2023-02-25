import numpy as np
import scipy.stats.mstats


def full_matmul_symmetrical(X, Y=None, correlation_type="pearson", avoid_copy=False, **kwargs):
    """
    Like `cor_matrix_asymmetrical` but incorporates the final adjustment 
    of the correlation matrix into the preprocessing step.
    """
    
    correlation_types = ["pearson", "spearman"]
    if correlation_type not in correlation_types:
        raise ValueError(f"Correlation type must be in: {correlation_types}")

    if correlation_type == "spearman":
        X = scipy.stats.mstats.rankdata(X, axis=0)

    # TODO: could be slightly optimized by reusing (x - mu) and dropping sqrt(m)
    XX = X - np.mean(X, axis=0)
    XX /= np.std(X, axis=0) * np.sqrt(X.shape[0])

    if Y is not None:
        if correlation_type == "spearman":
            Y = scipy.stats.mstats.rankdata(Y, axis=0)
        YY = Y - np.mean(Y, axis=0)
        YY /= np.std(Y, axis=0) * np.sqrt(X.shape[0])
    else:
        if avoid_copy:
            YY = XX
        else:
            # a copy is necessary here to speed up matmul when parallelizing; TODO: why?
            YY = XX.copy() 

    return np.matmul(XX.transpose(), YY)


def full_matmul_asymmetrical(X, Y=None, correlation_type="pearson", **kwargs):
    """
    The mean is only subtracted for the first matrix.
    """

    if correlation_type not in ["pearson", "spearman"]:
        raise ValueError(f"Correlation type must be in: {correlation_type}")

    if correlation_type == "spearman":
        X = scipy.stats.mstats.rankdata(X, axis=0)
        if Y is not None:
            Y = scipy.stats.mstats.rankdata(Y, axis=0)

    std = np.std(X, axis=0)
    
    XX = X - np.mean(X, axis=0)
    XX /= std * X.shape[0]
    
    if Y is not None:
        YY = Y
        YY /= np.std(Y, axis=0)
    else:
        YY = X / std
        
    return np.matmul(XX.transpose(), YY)
