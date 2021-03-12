import typing as tp
from functools import partial

import jax
import jax.numpy as jnp

from spax.sparse import COO, SparseArray
from spax.utils import (
    canonicalize_axis,
    multiply_leading_dims,
    segment_max,
    segment_softmax,
)


def matmul(mat: COO, v) -> jnp.ndarray:
    # assert mat.ndim == 2
    # return jax.vmap(lambda m, vi: m.matvec(vi), (None, 1), 1)(mat, v)
    v = jnp.asarray(v)
    rows, cols = mat.coords
    dv = multiply_leading_dims(mat.data, v[cols])
    out = jnp.zeros((mat.shape[0], *v.shape[1:]), dtype=dv.dtype)
    at = out.at[rows]
    return at.add(dv)
    return jnp.zeros((mat.shape[0], *v.shape[1:]), dtype=dv.dtype).at[rows].add(dv)


def transpose(mat: COO, axes=None) -> COO:
    if axes is None:
        axes = tuple(range(mat.ndim - 2)) + (-1, -2)
    assert len(axes) == mat.ndim
    return COO(
        mat.coords[jnp.asarray(axes)], mat.data, tuple(mat.shape[a] for a in axes)
    )


def reorder_perm(coords: jnp.ndarray, shape) -> jnp.ndarray:
    index1d = jnp.ravel_multi_index(coords, shape, mode="clip")
    return jnp.argsort(index1d)


def reorder(mat: COO) -> COO:
    """Reorder COO in ascending order of ravelled indices. Does not sum duplicates."""
    perm = reorder_perm(mat.coords, mat.shape)
    return COO(mat.coords[:, perm], mat.data[perm], mat.shape)


def add_coo_(x1: COO, x2: COO) -> COO:
    """Sum of two COO matrices, output in non-standard form."""
    assert x1.shape == x2.shape
    return COO(
        jnp.concatenate((x1.coords, x2.coords), axis=1),
        jnp.concatenate((x1.data, x2.data)),
        shape=x1.shape,
    )


def add_coo(x1: COO, x2: COO) -> COO:
    """
    Sum of two COO matrices, reordered to standard form.

    Duplicates are not removed. This allows it to be used in a jit context.
    """
    return reorder(add_coo_(x1, x2))


def with_data(mat: COO, data: jnp.ndarray) -> COO:
    assert mat.data.shape == data.shape, (mat.data.shape, data.shape)
    return COO(mat.coords, data, mat.shape)


def add(
    mat: COO, other: tp.Union[COO, int, float, jnp.ndarray]
) -> tp.Union[COO, jnp.ndarray]:
    if isinstance(other, SparseArray):
        return add_coo(mat, other.tocoo())
    other = jnp.asarray(other)
    if other.ndim == 0:
        return with_data(mat, mat.data + other)
    if other.ndim < mat.ndim:
        other = jnp.broadcast_to(other, mat.shape)
    row, col = mat.coords
    return other.at[..., row, col].add(mat.data)


def mul(mat: COO, other: tp.Union[int, float, jnp.ndarray]) -> COO:
    """Element-wise product, potentially with broadcasting."""
    other = jnp.asarray(other)
    if other.ndim == 0:
        return with_data(mat, mat.data * other)
    row, col = mat.coords
    if other.ndim == 1:
        return with_data(mat, mat.data * other[col])
    if other.ndim == 2:
        return with_data(mat, mat.data * other[row, col])
    raise NotImplementedError(
        f"other.ndim must be <= 2 but other has shape {other.shape}"
    )


def standardize(mat: COO) -> COO:
    """
    Reduce to standard form by sorting indices and summing duplicates.

    Not particularly performant. Not jitable. If indices are known to be unique, use
    `reorder` instead.
    """
    indices = jnp.ravel_multi_index(mat.coords, mat.shape)
    indices, inverse = jnp.unique(indices, return_inverse=True)
    data = jax.ops.segment_sum(mat.data, inverse, indices_are_sorted=False)
    valid = data != 0
    data = data[valid]
    indices = indices[valid]
    coords = jnp.unravel_index(indices, mat.shape)
    return COO(coords, data, shape=mat.shape)


def symmetrize(mat: COO) -> COO:
    """Get `(coo + coo.T) / 2`."""
    assert len(mat.shape) == 2 and mat.shape[-2] == mat.shape[-1], mat.shape
    mat = add_coo(mat, transpose(mat))
    return with_data(mat, mat.data / 2)


def symmetrize_data(mat: COO) -> COO:
    """
    Get `(mat + mat.T) / 2` assuming mat has symmetric sparsity.

    Symmetric sparsity pattern is not checked.
    """
    assert len(mat.shape) == 2 and mat.shape[-2] == mat.shape[-1], mat.shape
    mat_t = transpose(mat)
    perm = reorder_perm(mat_t.coords, mat_t.shape)
    return with_data(mat, (mat.data + mat.data[perm]) / 2)


def masked_inner(mat: COO, x, y) -> jnp.ndarray:
    """Comput `(x.T @ y)[row, col]`."""
    assert x.ndim == 2, x.shape
    assert y.ndim == 2, y.shape
    assert mat.ndim == 2, mat.shape
    row, col = mat.coords
    return (x[:, row] * y[:, col]).sum(axis=0)


def masked_outer(mat: COO, x, y) -> jnp.ndarray:
    """Compute `(x @ y.T)[row, col]`."""
    assert mat.ndim == 2, mat.shape
    row, col = mat.coords
    if x.ndim == 1:
        assert y.ndim == 1, (x.shape, y.shape)
        return x[row] * y[col]
    elif x.ndim == 2:
        assert y.ndim == 2, (x.shape, y.shape)
        assert (x.shape[0], y.shape[0]) == mat.shape and x.shape[1] == y.shape[1], (
            x.shape,
            y.shape,
            mat.shape,
        )
        return (x[row] * y[col]).sum(axis=1)
    raise ValueError(f"x and y must each be rank 1 or 2, got {x.shape}")


def masked_data(mat: COO, x: jnp.ndarray) -> jnp.ndarray:
    assert mat.ndim == x.ndim == 2, (mat.shape, x.shape)
    row, col = mat.coords
    return x[row, col]


def scale_rows(mat: COO, x: jnp.ndarray) -> COO:
    return with_data(mat, mat.data * x[mat.coords[0]])


def scale_columns(mat: COO, x: jnp.ndarray) -> COO:
    return with_data(mat, mat.data * x[mat.coords[1]])


def negate(mat: COO) -> COO:
    return with_data(mat, -mat.data)


def subtract(mat: COO, other) -> COO:
    if isinstance(other, SparseArray):
        return add_coo(mat, negate(other.tocoo()))
    return add(mat, -other)


def _reduce(mat: COO, axis, segment_reduction: tp.Callable) -> jnp.ndarray:
    if mat.ndim != 2:
        raise NotImplementedError("TODO")
    if not isinstance(axis, int):
        raise NotImplementedError("TODO")
    axis = canonicalize_axis(axis, mat.ndim)
    if axis == 0:
        not_axis = 1
    elif axis == 1:
        not_axis = 0
    else:
        raise NotImplementedError("TODO")
    return segment_reduction(
        mat.data, mat.coords[not_axis], num_segments=mat.shape[not_axis]
    )


def sum(mat: COO, axis=None) -> jnp.ndarray:
    if axis is None:
        return mat.data.sum()
    return _reduce(mat, axis, jax.ops.segment_sum)


def max(mat: COO, axis=None) -> jnp.ndarray:
    if axis is None:
        return mat.data.max()
    return _reduce(mat, axis, partial(segment_max, initial=0))


def _boolean_mask(
    mat: COO, mask: jnp.ndarray, valid_indices: jnp.ndarray, axis: int
) -> COO:
    assert jnp.issubdtype(mask.dtype, jnp.bool_)
    valid = mask[mat.coords[axis]]
    coords = mat.coords[:, valid]
    remapped = (
        jnp.zeros((mat.shape[axis],), dtype=mat.coords.dtype)
        .at[valid_indices]
        .set(jnp.arange(valid_indices.size))
    )
    coords = coords.at[axis].set(remapped[coords[axis]])
    data = mat.data[valid]
    shape = tuple(
        valid_indices.size if i == axis else s for i, s in enumerate(mat.shape)
    )
    return COO(coords, data, shape)


def boolean_mask(mat: COO, mask: jnp.ndarray, axis: int = 0):
    mask = jnp.asarray(mask, dtype=bool)
    assert mask.ndim == 1, mask.shape
    axis = canonicalize_axis(axis, mat.ndim)
    (valid_indices,) = jnp.where(mask)
    return _boolean_mask(mat, mask, valid_indices, axis)


def gather(mat: COO, indices: jnp.ndarray, axis: int = 0):
    axis = canonicalize_axis(axis, mat.ndim)
    mask = jnp.zeros((mat.shape[axis],), bool).at[indices].set(True)
    return _boolean_mask(mat, mask, indices, axis)


def softmax(mat: COO, axis: int = -1) -> COO:
    return with_data(mat, segment_softmax(mat.data, mat.coords[axis], mat.shape[axis]))


def to_coo(coo: COO) -> COO:
    return coo


def get_coords(mat: COO) -> jnp.ndarray:
    return mat.coords
