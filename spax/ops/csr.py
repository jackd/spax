import typing as tp

import jax.numpy as jnp

from spax.ops import coo as coo_lib
from spax.sparse import CSR
from spax.utils import multiply_leading_dims


def matmul(mat: CSR, v) -> jnp.ndarray:
    assert mat.ndim == 2
    v = jnp.asarray(v)
    dv = multiply_leading_dims(mat.data, v[mat.indices])
    ind = jnp.cumsum(jnp.zeros_like(mat.indices).at[mat.indptr].add(1))
    return jnp.zeros((mat.shape[0], *v.shape[1:]), dv.dtype).at[ind - 1].add(dv)


def transpose(mat: CSR, axes=None) -> CSR:
    return coo_lib.transpose(mat.tocoo(), axes=axes).tocsr()


def add(mat: CSR, other) -> tp.Union[CSR, jnp.ndarray]:
    out = coo_lib.add(mat.tocoo(), other)
    if isinstance(out, jnp.ndarray):
        return out
    return out.tocsr()


def mul(mat: CSR, other) -> CSR:
    return coo_lib.mul(mat.tocoo(), other).tocsr()


def symmetrize(mat: CSR) -> CSR:
    return coo_lib.symmetrize(mat.tocoo()).tocsr()


def rows(indptr: jnp.ndarray, dtype=jnp.int32, nnz: tp.Optional[int] = None):
    return _repeated_rows(jnp.arange(indptr.size - 1, dtype=dtype), indptr, nnz=nnz)


def _repeated_rows(
    x: jnp.ndarray, indptr: jnp.ndarray, axis=0, nnz: tp.Optional[int] = None
):
    if nnz is None:
        nnz = indptr[-1]
    return jnp.repeat(x, indptr[1:] - indptr[:-1], axis=axis, total_repeat_length=nnz)


def masked_inner(mat: CSR, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute (x.T @ y)[row, col] where (row, col) are the implied nonzero indices."""
    assert x.ndim == 2
    assert y.ndim == 2
    return (_repeated_rows(x, mat.indptr, axis=1, nnz=mat.nnz) * y[:, mat.indices]).sum(
        axis=0
    )


def masked_outer(mat: CSR, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute (x @ y.T)[row, col] where (row, col) are the nonzero indices."""
    assert mat.ndim == 2, mat.shape
    assert x.ndim == 1, x.shape
    assert y.ndim == 1, y.shape
    return _repeated_rows(x, mat.indptr, nnz=mat.nnz) * y[mat.indices]


def with_data(mat: CSR, data: jnp.ndarray) -> CSR:
    assert mat.data.shape == data.shape, (mat.shape, data.shape)
    return CSR(mat.indices, mat.indptr, data, shape=mat.shape)


def masked_data(mat: CSR, x: jnp.ndarray) -> jnp.ndarray:
    return coo_lib.masked_data(mat.tocoo(), x)


def scale_rows(mat: CSR, x: jnp.ndarray) -> CSR:
    return with_data(mat, mat.data * _repeated_rows(x, mat.indptr, nnz=mat.nnz))


def scale_columns(mat: CSR, x: jnp.ndarray) -> CSR:
    return with_data(mat, mat.data * x[mat.indices])
