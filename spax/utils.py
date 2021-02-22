import inspect
import typing as tp
from functools import wraps

import jax
import jax.numpy as jnp

from spax import sparse


def multiply_leading_dims(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    assert b.ndim >= a.ndim
    assert a.shape == b.shape[: a.ndim]
    return a.reshape(*a.shape, *(1,) * (b.ndim - a.ndim)) * b


def eye(
    N: int, dtype: jnp.dtype = jnp.float32, index_dtype: jnp.dtype = jnp.int32
) -> sparse.COO:
    return diag(jnp.ones((N,), dtype=dtype), index_dtype=index_dtype)


def diag(diagonals: jnp.ndarray, index_dtype: jnp.dtype = jnp.int32) -> sparse.COO:
    """Create a matrix with `diagonals` on the main diagonal."""
    assert diagonals.ndim == 1
    n = diagonals.size
    r = jnp.arange(n, dtype=index_dtype)
    return sparse.COO(jnp.vstack((r, r)), diagonals, (n, n))


_FMTS = {"bsr": sparse.BSR, "coo": sparse.COO, "csr": sparse.CSR, "ell": sparse.ELL}


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

        if fmt == "dense":
            return mat
        elif fmt in _FMTS:
            return _FMTS[fmt].fromdense(mat)
        else:
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
    fmt: str = "csr",
) -> sparse.SparseArray:
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
) -> sparse.SparseArray:
    """Sparse Normal Array"""
    return jax.random.normal(key, shape=shape, dtype=dtype)


def non_negative_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    assert axis < ndim, (axis, ndim)
    return axis
