import os
import sys
import warnings


def set_threads_for_external_libraries(n_threads=1):
    """
    Tries to disable BLAS or similar implicit  parallelization engines
    by setting the following environment attributes to `1`:

    - OMP_NUM_THREADS
    - OPENBLAS_NUM_THREADS
    - MKL_NUM_THREADS
    - VECLIB_MAXIMUM_THREADS
    - NUMEXPR_NUM_THREADS

    This can be useful since `numpy` and co. are using these libraries to parallelize vector and matrix operations.
    However, by default, the grab ALL CPUs an a machine. Now, if you use parallelization, e.g., based on `joblib` as in
    `sklearn.model_selection.GridSearchCV` or `sklearn.model_selection.cross_validate`, then you will overload your
    machine and the OS scheduler will spend most of it's time switching contexts instead of calculating.

    Parameters
    ----------
    n_threads: int, optional, default: 1
        Number of threads to use for

    Notes
    -----
    - This ONLY works if you import this file BEFORE `numpy` or similar libraries.
    - BLAS and co. only kick in when the data (matrices etc.) are sufficiently large.
      So you might not always see the `CPU stealing` behavior.
    - For additional info see: "2.4.7 Avoiding over-subscription of CPU ressources" in the
      `joblib` docs (https://buildmedia.readthedocs.org/media/pdf/joblib/latest/joblib.pdf).
    - Also note that there is a bug report for `joblib` not disabling BLAS and co.
      appropriately: https://github.com/joblib/joblib/issues/834

    Returns
    -------
    None

    """

    if (
            "numpy" in sys.modules.keys()
            or "scipy" in sys.modules.keys()
            or "sklearn" in sys.modules.keys()
            ):
        warnings.warn("This function should be called before `numpy` or similar modules are imported.")

    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
