import typing as tp

import jax
import jax.numpy as jnp
from jax.experimental.sparse_ops import COO, CSR

import spax.ops.coo as coo_lib
from spax.utils import canonicalize_axis


def to_coo(csr: CSR) -> COO:
    row = (
        jnp.cumsum(jnp.zeros_like(csr.indptr, shape=csr.nnz).at[csr.indptr].add(1)) - 1
    )
    return COO((csr.data, row, csr.indices), shape=csr.shape)


def mul(mat: CSR, other) -> CSR:
    return coo_lib.to_csr(coo_lib.mul(to_coo(mat), other))


def symmetrize(mat: CSR) -> CSR:
    return coo_lib.to_csr(coo_lib.symmetrize(to_coo(mat)))


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
    return CSR((data, mat.indices, mat.indptr), shape=mat.shape)


def masked_data(mat: CSR, x: jnp.ndarray) -> jnp.ndarray:
    return coo_lib.masked_data(to_coo(mat), x)


def scale_rows(mat: CSR, x: jnp.ndarray) -> CSR:
    return with_data(mat, mat.data * _repeated_rows(x, mat.indptr, nnz=mat.nnz))


def scale_columns(mat: CSR, x: jnp.ndarray) -> CSR:
    return with_data(mat, mat.data * x[mat.indices])


def sum(mat: CSR, axis: tp.Optional[int] = None):  # pylint: disable=redefined-builtin
    if axis is None:
        return mat.data.sum()
    axis = canonicalize_axis(axis, 2)
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
    return coo_lib.to_csr(coo_lib.softmax(to_coo(mat), axis=axis))


def symmetrize_data(mat: CSR) -> CSR:
    return with_data(mat, coo_lib.symmetrize_data(to_coo(mat)).data)


def get_coords(mat: CSR) -> jnp.ndarray:
    return jnp.stack(
        (rows(mat.indptr, dtype=mat.indices.dtype, nnz=mat.nnz), mat.indices), axis=0
    )


def to_csr(mat: CSR) -> CSR:
    return mat
