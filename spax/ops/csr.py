import typing as tp

import jax
import jax.numpy as jnp

from spax.ops import coo as coo_lib
from spax.sparse import CSR
from spax.utils import canonicalize_axis


def _matvec_components(indices, indptr, data, v):
    dv = data * v[indices]
    size = indptr.size - 1
    ind = jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1))
    return jnp.zeros(size, dv.dtype).at[ind - 1].add(dv)


def matmul(mat: CSR, v) -> jnp.ndarray:
    # HACK
    assert mat.ndim == 2
    v = jnp.asarray(v)
    if v.ndim == 1:
        return _matvec_components(mat.indices, mat.indptr, mat.data, v)
    assert v.ndim == 2
    return jax.vmap(_matvec_components, (None, None, None, 1), 1)(
        mat.indices, mat.indptr, mat.data, v
    )
    # dv = multiply_leading_dims(mat.data, v[mat.indices])
    # ind = jnp.cumsum(jnp.zeros_like(mat.indices).at[mat.indptr].add(1))
    # return jnp.zeros((mat.shape[0], *v.shape[1:]), dv.dtype).at[ind - 1].add(dv)


def transpose(mat: CSR, axes=None) -> CSR:
    return coo_lib.transpose(mat.tocoo(), axes=axes).tocsr()


def add(mat: CSR, other) -> tp.Union[CSR, jnp.ndarray]:
    out = coo_lib.add(mat.tocoo(), other)
    if hasattr(out, "tocsr"):
        return out.tocsr()
    # dense array
    return out


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
    return jnp.repeat(x, jnp.diff(indptr), axis=axis, total_repeat_length=nnz)


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
    assert x.ndim == y.ndim, (x.shape, y.shape)
    xr = _repeated_rows(x, mat.indptr, nnz=mat.nnz)
    yc = y[mat.indices]
    out = xr * yc
    if x.ndim == 1:
        return out

    assert x.ndim == 2, x.shape
    return out.sum(axis=1)


def with_data(mat: CSR, data: jnp.ndarray) -> CSR:
    assert mat.data.shape == data.shape, (mat.data.shape, data.shape)
    return CSR(mat.indices, mat.indptr, data, shape=mat.shape)


def masked_data(mat: CSR, x: jnp.ndarray) -> jnp.ndarray:
    return coo_lib.masked_data(mat.tocoo(), x)


def scale_rows(mat: CSR, x: jnp.ndarray) -> CSR:
    return with_data(mat, mat.data * _repeated_rows(x, mat.indptr, nnz=mat.nnz))


def scale_columns(mat: CSR, x: jnp.ndarray) -> CSR:
    return with_data(mat, mat.data * x[mat.indices])


def sum(mat: CSR, axis: tp.Optional[int] = None):
    if axis is None:
        return mat.data.sum()
    axis = canonicalize_axis(axis, mat.ndim)
    if axis == 0:
        segment_ids = mat.indices
        indices_are_sorted = False
        num_segments = mat.shape[1]
    elif axis == 1:
        segment_ids = rows(mat.indptr, nnz=mat.nnz)
        indices_are_sorted = True
        num_segments = mat.shape[0]
    else:
        raise ValueError(f"cannonicalized axis must be 0 or 1, got {axis}")
    return jax.ops.segment_sum(
        mat.data,
        segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
    )


def softmax(mat: CSR, axis: int = -1) -> CSR:
    return coo_lib.softmax(mat.tocoo(), axis=axis).tocsr()


def to_coo(mat: CSR) -> coo_lib.COO:
    row = jnp.repeat(
        jnp.arange(mat.shape[0]), jnp.diff(mat.indptr), total_repeat_length=mat.nnz,
    )
    col = mat.indices
    return coo_lib.COO(jnp.vstack([row, col]), mat.data, mat.shape)


def symmetrize_data(mat: CSR) -> CSR:
    return with_data(mat, coo_lib.symmetrize_data(to_coo(mat)).data)


def get_coords(mat: CSR) -> jnp.ndarray:
    return jnp.stack(
        (rows(mat.indptr, dtype=mat.indices.dtype, nnz=mat.nnz), mat.indices), axis=0
    )
