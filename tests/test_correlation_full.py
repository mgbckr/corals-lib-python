import numpy as np


def create_data():
    X = np.random.random((5, 10))
    return X


def test_corrcoef():
    from corals.correlation.full.baselines import full_corrcoef
    X = create_data()
    cor_default_values = full_corrcoef(X)
    assert cor_default_values.shape[0] == X.shape[1] 


def test_cor_loop():
    from corals.correlation.full._deprecated.original import full_loop
    default_test_cor(full_loop)


def test_cor_matmul_summetrical():
    from corals.correlation.full.matmul import full_matmul_symmetrical
    default_test_cor(full_matmul_symmetrical)


def default_test_cor(cor_function):
    
    from corals.correlation.full.baselines import full_corrcoef

    X = create_data()
    cor_test_values = cor_function(X)
    assert cor_test_values.shape[0] == X.shape[1] 
    assert np.allclose(cor_function(X), cor_function(X, X))
    
    cor_default_values = full_corrcoef(X)
    assert np.allclose(cor_default_values, cor_test_values)
