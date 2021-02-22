import typing as tp

import jax.numpy as jnp

from spax import sparse

ArrayFun = tp.Callable[[jnp.ndarray], jnp.ndarray]
ArrayOrFun = tp.Union[ArrayFun, jnp.ndarray, sparse.SparseArray]
