import re
import numpy as np
import scipy.stats


def preprocess_X(X):
    Xh = X - np.mean(X, axis=0)
    Xh /= np.std(Xh, axis=0) * np.sqrt(X.shape[0])
    return Xh


def preprocess_XY(X, Y=None, correlation_type="pearson", normalize=True, Y_fill_none=False, ensure_copy="none"):

    if not Y_fill_none and Y is not None:
        raise ValueError("Specifying Y is not supported.")

    # rank data for spearman
    if correlation_type == "pearson":
        pass
    elif correlation_type == "spearman":
        X = scipy.stats.mstats.rankdata(X, axis=0)
        if Y is not None:
            Y = scipy.stats.mstats.rankdata(Y, axis=0)
    else:
        raise ValueError("Correlation types other than 'pearson' and 'spearman' not supported.")

    # normalize and fill Y
    Xh = preprocess_X(X) if normalize else X
    if Y is None:
        Yh = Xh if Y_fill_none else None
    else:
        Yh = preprocess_X(Y) if normalize else Y

    # ensure copy of X and Y
    if correlation_type == "pearson" and not normalize:
        if not re.match("none|[XyY]+", ensure_copy):
            raise ValueError(f"Invalid copy mode: '{ensure_copy}'")
        if "X" in ensure_copy:
            Xh = Xh.copy()
        if "Y" in ensure_copy or ("y" in ensure_copy and Y is None):
            Yh = Yh.copy()        

    return Xh, Yh


def argsort_topk(a, k):
    partition = np.argpartition(a, kth=k)
    return partition[:k][np.argsort(a[partition][:k])]


def argtopk(
        values, 
        threshold=None, 
        k=None, 
        argtopk_method="argpartition", 
        require_sorted_topk=True, 
        return_values=False):

    if values.ndim > 1:
        raise ValueError("Only one dimensional arrays are supported.")

    if k is None:
        k = values.size
    else:
        k = min(k, values.size)

    # threshold values
    if threshold is not None:
        threshold_idx = np.argwhere(values < threshold).flatten()
        values = values[threshold_idx]
        k = min(k, values.size)
    
    # print(values)
    # print(values.size, k, threshold)

    # argsort values

    if isinstance(argtopk_method, str):

        if argtopk_method == "argsort":
            topk_idx = np.argsort(values)[:k]

        elif argtopk_method == "argpartition":

            if k < values.size:
                topk_idx = np.argpartition(values, kth=k - 1)[:k]

                if require_sorted_topk:
                    partition = values[topk_idx]
                    topk_idx = topk_idx[np.argsort(partition)]
            
            elif require_sorted_topk:
                topk_idx = np.argsort(values)
                
            else:
                topk_idx = np.arange(values.size)

        else:
            raise ValueError(f"Unknown sorting function: {argtopk_method}")

    elif callable(argtopk_method):
        topk_idx = argtopk_method(values, k, require_sorted_topk)

    else:
        raise ValueError(f"Unknown sorting function: {argtopk_method}")

    # convert index on thresholded values back to original index 
    topk_values = values[topk_idx]
    if threshold is not None:
        topk_idx = threshold_idx[topk_idx]

    # return values
    if return_values:
        return topk_idx, topk_values 
    else:
        return topk_idx


def derive_k(X, Y, k=None, handle_none='forward'):
    """
    Derive k.
    """

    # handle none
    if k is None:
        if handle_none == "forward":
            return None
        else:
            k = handle_none

    # make sure Y is set
    if Y is None:
        Y = X
    
    # handle special k definitions
    if isinstance(k, str):
        if k == "all":
            k = X.shape[1] * Y.shape[1]
        else:
            raise ValueError(f"Unknown value for k: {k}")
    elif k < 1:
        k = X.shape[1] * Y.shape[1] * k

    return int(np.ceil(k))



def derive_k_per_query(X, Y, k, approximation_factor):
    """
    Helper.
    """

    if approximation_factor is None:
        return min(k, X.shape[1])

    if approximation_factor < 1:
        raise ValueError(
            f"Approximation factor must be >= 1; it was: {approximation_factor}")
    
    return int(min(
        int(np.ceil(k / Y.shape[1])) * approximation_factor, 
        X.shape[1]))


def derive_k_per_batch(X, Y, n_batches, k, approximation_factor=None):

    if approximation_factor is None:
        return k

    elif approximation_factor < 1:
        raise ValueError(
            f"Approximation factor must be >= 1 or None; it was: {approximation_factor}")

    else:
        return int(min(
            int(np.ceil(k / n_batches)) * approximation_factor, 
            X.shape[1] * Y.shape[1]))


def derive_topk(
        mask_inverse, 
        dst, 
        idx_r, 
        idx_c, 
        k,
        threshold=None, 
        argtopk_method="argsort", 
        require_sorted_topk=True):
    
    # calculate correlations
    cor = 1 - dst**2 / 2
    
    # sort
    cor_order = argtopk(
        -cor, 
        threshold=-threshold if threshold is not None else None, 
        k=k, 
        argtopk_method=argtopk_method, 
        require_sorted_topk=require_sorted_topk, 
        return_values=False)

    # fix correlations from -Xh
    if mask_inverse is not None:
        cor[mask_inverse] *= -1

    return cor[cor_order][:k], (idx_r[cor_order][:k], idx_c[cor_order][:k])


def derive_bins(n, n_batches):
    n_batches = max(1, min(n_batches, n - 2))
    bin_size_default = n // n_batches
    bin_sizes = np.repeat(bin_size_default, n_batches)
    for i in range(n - bin_size_default * n_batches):
        bin_sizes[i % len(bin_sizes)] += 1
    bins = np.insert(np.cumsum(bin_sizes), 0, 0)
    return bins


def derive_pvalues(correlations, n_samples):
    """
    Calculate pvalues from correlations given the number of samples used to calculate the correlations.
    Source: https://stackoverflow.com/a/24547964/991496
    """
    correlations = np.asarray(correlations)
    rf = correlations
    df = n_samples - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = scipy.special.betainc(0.5 * df, 0.5, df / (df + ts))
    return pf


def multiple_test_correction(pvalues, n_features, method="bonferroni", minimal_pvalues=False, alpha_level=0.05):
    """Calculates adjusted pvalues for calculated correlations based on the number of features.

    Parameters
    ----------
    pvalues : np.array
        pvalues
    n_features : int
        number of features
    method : str, optional
        multiple test correction method (`bonferroni` or `fdr_bh`), by default "bonferroni"
    minimal_pvalues : bool, optional
        can be set to `true` if the pvalues for the diagonal and duplicate correlations have been removed, by default False
    alpha_level: numeric, optional
        alpha level used in some tests ("fdr_bh"), by default 0.05
    """
        
    if minimal_pvalues:
        n_ref = (n_features**2 - n_features) / 2
    else:
        n_ref = n_features**2 
        
    if method == "bonferroni":
        adjusted_sorted = pvalues * n_ref
        adjusted_sorted[adjusted_sorted > 1] = 1
        return adjusted_sorted
    
    elif method == "fdr_bh":
        order = np.argsort(pvalues)
        pvalues_sorted = pvalues[order]
        adjusted_sorted = pvalues_sorted * n_ref / np.arange(1, len(pvalues_sorted) + 1)
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted[adjusted_sorted > 1] = 1
        
        # truncation
        last_value = adjusted_sorted[-1]
        adjusted_overhang = 1 * n_ref / len(pvalues_sorted) + 1
        if last_value > adjusted_overhang:
            adjusted_sorted[adjusted_sorted == last_value] = adjusted_overhang
        
        adjusted = np.empty(len(adjusted_sorted))
        adjusted[order] = adjusted_sorted
        
        return adjusted

    else:
        raise ValueError(f"Unknown correction method: {method}")
