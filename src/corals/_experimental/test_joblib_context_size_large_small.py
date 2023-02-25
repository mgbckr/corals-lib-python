
from memory_profiler import memory_usage


def run(n_jobs=64, context="large", preferred_backend=None, memory_backend="psutil_uss"):
    import time
    import memory_profiler
    import joblib

    if context == "large":
        import sklearn
        import numba
        import numpy as np
        import corals.correlation.topk.base
    elif context == "small":
        pass

    def job():
        time.sleep(10)

    def test():
        results = joblib.Parallel(n_jobs=n_jobs, prefer=preferred_backend)(
            joblib.delayed(job)()
            for i in range(n_jobs))

    memory_usage_kwargs = dict(
        # doesn't make a difference either it seems
        # default value (0.1) seems fine
        interval=0.1,
        # we are only interested in max imum memory consumption over time
        max_usage=True,  
        # combines memory usage of parent and children processes
        # NOTE: This measures RSS and might overestimate memory usage! 
        #       There is an updated version of `memory_profiler` coming up 
        #       that can measure PSS and USS which might be more accurate 
        include_children=True,
        # also keep track of children's memory consumption separately ... we don't really use this
        multiprocess=True,
        # default backend measures RSS which may overestimate memory usage in parallel case
        backend=memory_backend
    )

    result = memory_profiler.memory_usage(proc=test, **memory_usage_kwargs)
    print(result)


if __name__ == '__main__':
    # run(context="small")
    run(context="large")

    # conclusion: not much of a difference ... I really thought I saw an effect like that ... need to explore this more at some point