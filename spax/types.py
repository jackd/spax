import typing as tp

import jax.numpy as jnp

from spax import sparse

ArrayFun = tp.Callable[[jnp.ndarray], jnp.ndarray]
ArrayOrFun = tp.Union[ArrayFun, jnp.ndarray, sparse.SparseArray]


class EigenPair(tp.NamedTuple):
    """Result of eigendecomposition, or a single eigenpair."""

    w: jnp.ndarray  # [...] eigenvalue
    v: jnp.ndarray  # [N, ...] eigenvector
