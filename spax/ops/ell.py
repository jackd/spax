import jax.numpy as jnp

from spax.sparse import ELL
from spax.utils import multiply_leading_dims


def matmul(mat: ELL, v) -> jnp.ndarray:
    assert mat.ndim == 2
    v = jnp.asarray(v)
    invalid = jnp.arange(mat.data.shape[1]) >= mat.rownz[:, None]
    dv = multiply_leading_dims(mat.data, v[mat.columns])
    return dv.at[invalid].set(0).sum(1, dtype=dv.dtype)
