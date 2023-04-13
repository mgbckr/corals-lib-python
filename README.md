# CorALS

*CorALS* is an open-source software package for the construction and analysis of large-scale correlation networks for high-dimensional data.

If you use *CorALS* for a scientific publication, please cite:

```plain
Becker, M., Nassar, H., Espinosa, C. et al. 
Large-scale correlation network construction for unraveling the coordination of complex biological systems. 
Nat Comput Sci (2023). 
https://doi.org/10.1038/s43588-023-00429-y
```

## Install

```bash
pip install corals
```

## Quick start

The following quick start examples can also be found in [an executable notebook](https://github.com/mgbckr/corals-lib-python/tree/main/docs/notebooks/quickstart.ipynb).

**Note:** If any of the following examples do not work, check the [previously mentioned executable notebook](https://github.com/mgbckr/corals-lib-python/tree/main/docs/notebooks/quickstart.ipynb) as well. It is tested automatically, and this `README` may not have been updated.

### Prepare parallelization

Before running anything, we make sure that `numpy` will not  oversubscribe CPUs and slow things down.
Note that this has to be executed **before importing `numpy`**.

* For full correlation matrix calculation, setting `n_threads > 1` can be used to parallelize the calculation.
* For the top-k approaches, setting `n_threads=1` makes the most sense, since parallelization is specified separately.

```python
from corals.threads import set_threads_for_external_libraries
set_threads_for_external_libraries(n_threads=1)
```

### Load data

Create some data (alternatively load your own):

```python
import numpy as np

# create random data
n_features = 20000
n_samples = 50
X = np.random.random((n_samples, n_features))
```

### Full correlation matrix computation

```python
# runtime: ~2 sec
from corals.correlation.full.base import cor_full
cor_values = cor_full(X)
```

### Top-k correlation matrix computation using Spearman correlation

```python
# runtime: ~5 sec with `n_jobs=8`
from corals.correlation.topk.base import cor_topk
cor_topk_result = cor_topk(X, k=0.001, correlation_type="spearman", n_jobs=8)
```

### Top-k differential correlation matrix computation using Spearman correlation

```python
# generate some more data
X1 = X
X2 = np.random.random((n_samples, n_features))
```

```python
# runtime: ~5 sec with `n_jobs=8`

from corals.correlation.topkdiff.base import cor_topkdiff
cor_topkdiff_result = cor_topkdiff(X1, X2, k=0.001, correlation_type="spearman", n_jobs=8)
```

### Calculating p-values

```python

# reusing correlation from the top-k example
# runtime: ~20 sec with `n_jobs=8`
from corals.correlation.topk.base import cor_topk
cor_topk_values, cor_topk_coo = cor_topk(X, correlation_type="spearman", k=0.001, n_jobs=8)

from corals.correlation.utils import derive_pvalues, multiple_test_correction
n_samples = X.shape[0]
n_features = X.shape[1]

# calculate p-values
pvalues = derive_pvalues(cor_topk_values, n_samples)

# multiple hypothesis correction
pvalues_corrected = multiple_test_correction(pvalues, n_features, method="fdr_bh")
```

## Detailed examples

For detailed examples and recommendations, see the corresponding [notebook](https://github.com/mgbckr/corals-lib-python/tree/main/docs/notebooks/full.ipynb).

The `docs/notebooks` folder may contain additional examples and tutorials in the form of Jupyter Notebooks.

Quick setup for Jupyter notebooks.

```bash
export ENV_NAME=corals

conda create -n ${ENV_NAME} python=3.10
conda activate ${ENV_NAME}
pip install corals

conda install -c conda-forge jupyterlab  # optional if Jupyter Lab is already installed
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name ${ENV_NAME}
```

## Development

**TODO**: add documentation for contributing new code / methods

### Setup

```bash
git clone git@github.com:mgbckr/corals-lib-python.git
pip install -e .
```

### Release

```bash
git tag -a x.x.x -m "Release x.x.x"
```
