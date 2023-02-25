import numpy as np
import numba

     
@numba.njit
def np_apply_along_axis(func1d, axis, arr):
    """https://github.com/numba/numba/issues/1269"""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty((1, arr.shape[1]), dtype=arr.dtype)
        for i in range(result.shape[1]):
            result[0, i] = func1d(arr[:, i])
    else:
        result = np.empty((arr.shape[0], 1), dtype=arr.dtype)
        for i in range(result.shape[0]):
            result[i, 0] = func1d(arr[i, :])
    return result


@numba.njit
def np_max(array, axis):
    return np_apply_along_axis(np.max, axis, array)


@numba.njit
def np_min(array, axis):
    return np_apply_along_axis(np.min, axis, array)


@numba.njit
def lexsort(a, b): 
    """
    Sorts by b first, then by a; analogously to:
    https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html

    This is implemented by hand rather than using "np.lexsort" because of: 
    https://github.com/numpy/numpy/issues/12755
    """
    idxs = np.argsort(a, kind="mergesort") 
    return idxs[np.argsort(b[idxs], kind="mergesort")]


@numba.njit
def sort_topk_idx(sorted_values, idx):
    """TODO: we could do this inplace."""
   
    sorted_values_s = sorted_values.copy()
    idx_s = np.concatenate((idx[0].reshape(1,-1), idx[1].reshape(1,-1)), axis=0)
    
    # idx_m = np.concatenate((
    #     np.max(idx_s, axis=0, keepdims=True), 
    #     np.min(idx_s, axis=0, keepdims=True)))

    idx_m = np.concatenate((
        np_min(idx_s, axis=0), 
        np_max(idx_s, axis=0)))
        
    # sort stretches of same values according to their indices
    idx_start = 0
    value = np.abs(sorted_values_s[0])
    for idx_end in range(len(sorted_values_s) + 1):
        if idx_end == len(sorted_values) or np.abs(sorted_values_s[idx_end]) != value:  ## TODO: change to np.close or something?
            if idx_end - idx_start > 2:
                order = lexsort(idx_m[1,idx_start:idx_end], idx_m[0,idx_start:idx_end])
    
                sorted_values_s[idx_start:idx_end] = sorted_values_s[idx_start:idx_end][order]
                idx_s[:,idx_start:idx_end] = idx_s[:,idx_start:idx_end][:,order]
            
            idx_start = idx_end
            if idx_end != len(sorted_values):
                value = np.abs(sorted_values_s[idx_start])
            
    return sorted_values_s, (idx_s[0,:], idx_s[1,:])
                            

@numba.njit
def symmetrize_topk(sorted_values, idx):

    idx_r, idx_c = idx
    sorted_values_sym = np.empty(len(sorted_values))
    idx_r_sym = np.empty(len(idx[0]), dtype=np.int64)
    idx_c_sym = np.empty(len(idx[1]), dtype=np.int64)

    i = 0
    i_sym = 0
    while i_sym < len(sorted_values) - 1:
        
        idx_ri, idx_rii = idx_r[i], idx_r[i + 1]
        idx_ci, idx_cii = idx_c[i], idx_c[i + 1]
        
        if idx_ri == idx_ci:
            sorted_values_sym[i_sym] = sorted_values[i]
            idx_r_sym[i_sym] = idx_ri
            idx_c_sym[i_sym] = idx_ci
            i += 1
            i_sym += 1
            
        elif (idx_ri == idx_cii) & (idx_ci == idx_rii):
            
            sorted_values_sym[i_sym] = sorted_values[i]
            sorted_values_sym[i_sym + 1] = sorted_values[i + 1]
            
            idx_r_sym[i_sym] = idx_ri
            idx_r_sym[i_sym + 1] = idx_rii
            
            idx_c_sym[i_sym] = idx_ci
            idx_c_sym[i_sym + 1] = idx_cii
            
            i += 2
            i_sym += 2
        else:
            sorted_values_sym[i_sym:i_sym+2] = sorted_values[i]
            
            idx_r_sym[i_sym] = idx_ri
            idx_r_sym[i_sym + 1] = idx_ci
            
            idx_c_sym[i_sym] = idx_ci
            idx_c_sym[i_sym + 1] = idx_ri
            
            i += 1
            i_sym += 2
            
    # uneven case
    if i_sym == len(sorted_values) - 1:
        sorted_values_sym[i_sym] = sorted_values[i]
        idx_r_sym[i_sym] = idx_r[i]
        idx_c_sym[i_sym] = idx_c[i]
            
    return sorted_values_sym, (idx_r_sym, idx_c_sym)


