# Write to disk

Writing to disk is generally not problem with the `ResultHandler`. However, there seems to be an issue with the current implementation if the final result should only appear on disk.

## Background

Currently the implementation responsibility is as follows:

* `TopkMapReduce` provides `prepare`, `map`, and `reduce`
* `ResultHandler` provides functionality to store results of `map` to be loaded by `reduce`.

The idea was to keep the Mapper and Reducer in one class since they know their internal data structure.
Additionally, the `ResultMapper` was supposed to uncouple the transmission of map results to the reduce function from the actual implementation of the `TopkMapReduce`.

In some cases the reducer has to know and potentially load *all* map results, e.g., when sorting top-k correlations. I am not sure how to circumvent this at this point without pushing final storage functionality into the reducer. **This could result in very specialized reducers, which could get complicated.**

## Potential solutions

TODO
