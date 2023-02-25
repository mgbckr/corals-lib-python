import numpy as np
import scipy.stats
import scipy.linalg


def full_loop(X, Y=None):
    if Y is None:
        Y = X
    cor = np.empty((X.shape[1], Y.shape[1]))
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            cor[i,j] = scipy.stats.pearsonr(X[:,i], Y[:,j])[0]
    return cor


def full_corrcoef(X, Y=None):
    if Y is not None:
        raise NotImplementedError("Specifying and independent Y is not supported.")
    return np.corrcoef(X, rowvar=False)


def full_matrix_asymmetrical(X, Y=None):
    """
    The mean is only subtracted for the first matrix.
    """
    
    std = np.std(X, axis=0)
    
    XX = X - np.mean(X, axis=0)
    XX /= std * X.shape[0]
    
    if Y is not None:
        YY = Y
        YY /= np.std(Y, axis=0)
    else:
        YY = X / std
        
    return np.matmul(XX.transpose(), YY)


def full_matrix_symmetrical(X, Y=None, avoid_copy=False, spearman=False):
    """
    Like `cor_matrix_asymmetrical` but incorporates the final adjustment 
    of the correlation matrix into the preprocessing step.
    """
    
    if spearman:
        X = scipy.stats.mstats.rankdata(X, axis=0)

    # TODO: could be slightly optimized by reusing (x - mu) and dropping sqrt(m)
    XX = X - np.mean(X, axis=0)
    XX /= np.std(X, axis=0) * np.sqrt(X.shape[0])

    if Y is not None:
        if spearman:
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
