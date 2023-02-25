import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# TODO: DO NOT DO THIS!! It prevents allowing to set threads via `corals.threads.set_threads_for_external_libraries`. Also it seems to load up the module space which is transferred for `joblib` workers which in turn increases the memory usage by a lot!
# from corals.correlation.full import full_matrix_symmetrical as cor_matrix
# from corals.correlation.topk import topk_balltree_combined_tree_parallel_optimized as cor_topk
# from corals.correlation.topkdiff import topkdiff_balltree_combined_tree_parallel as cor_topkdiff
