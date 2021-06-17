import inspect
import typing as tp
from functools import wraps

import jax
import jax.numpy as jnp
from jax.experimental.sparse_ops import COO, CSR, JAXSparse

from spax.ops import to_csr

_FMTS = {"coo": COO, "csr": CSR}


def _sparse_rng(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        sig = inspect.signature(func).bind(*args, **kwargs)
        sig.apply_defaults()
        arguments = dict(sig.arguments)
        key = arguments.pop("key")
        nnz = arguments["nnz"]
        fmt = arguments["fmt"]

        # TODO(jakevdp): avoid creating dense array.
        key, key2 = jax.random.split(key)
        mat = func(key, **arguments)
        nnz = int(nnz * mat.size if 0 < nnz < 1 else nnz)

        if nnz <= 0:
            mat = jnp.zeros_like(mat)
        elif nnz < mat.size:
            mask = jax.random.shuffle(key2, jnp.arange(mat.size)).reshape(mat.shape)
            mat = jnp.where(mask < nnz, mat, 0)

        if fmt is jnp.ndarray:
            return mat
        mat = COO.fromdense(mat)
        if fmt is COO:
            return mat
        if fmt is CSR:
            return to_csr(mat)
        # if fmt in _FMTS:
        #     return _FMTS[fmt].fromdense(mat)
        raise ValueError(f"Unrecognized format: {fmt}")

    return wrapped


@_sparse_rng
def random_uniform(
    key: jnp.ndarray,
    shape: tp.Sequence[int] = (),
    dtype: jnp.dtype = jnp.float32,
    minval: tp.Union[float, jnp.ndarray] = 0.0,
    maxval: tp.Union[float, jnp.ndarray] = 1.0,
    nnz: tp.Union[int, float] = 0.1,
    fmt: type = CSR,
) -> JAXSparse:
    """Sparse Uniform Array"""
    return jax.random.uniform(
        key, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )


@_sparse_rng
def random_normal(
    key: jnp.ndarray,
    shape: tp.Sequence[int] = (),
    dtype: jnp.dtype = jnp.float32,
    nnz: tp.Union[int, float] = 0.1,
    fmt: str = "csr",
) -> JAXSparse:
    """Sparse Normal Array"""
    return jax.random.normal(key, shape=shape, dtype=dtype)
