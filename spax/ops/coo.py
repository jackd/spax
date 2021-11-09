import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config as jax_config
from jax.experimental.sparse.ops import COO, CSR

from spax.utils import canonicalize_axis, segment_max, segment_softmax


def indices_1d(row: jnp.ndarray, col: jnp.ndarray, shape) -> jnp.ndarray:
    if np.prod(np.asarray(shape, dtype=np.int64)) > np.iinfo(np.int32).max:
        if jax_config.x64_enabled:
            row = row.astype(jnp.int64)
            col = col.astype(jnp.int64)
            shape = jnp.asarray(shape, dtype=jnp.int64)
        else:
            raise ValueError(
                "Overflow likely. Enable x64 for this operation with "
                "`jax.experimental.enable_x64` context."
            )
    return jnp.ravel_multi_index(
        (row, col),
        shape,
        mode="clip",
    )


def reorder_perm(row: jnp.ndarray, col: jnp.ndarray, shape) -> jnp.ndarray:
    return jnp.argsort(indices_1d(row, col, shape))


def reorder(mat: COO) -> COO:
    """Reorder COO in ascending order of ravelled indices. Does not sum duplicates."""
    perm = reorder_perm(mat.row, mat.col, mat.shape)
    return COO((mat.data[perm], mat.row[perm], mat.col[perm]), shape=mat.shape)


def add_coo_(x1: COO, x2: COO, validate_shape=True) -> COO:
    """Sum of two COO matrices, output in non-standard form."""
    if validate_shape:
        assert x1.shape == x2.shape, (x1.shape, x2.shape)
    return COO(
        (
            jnp.concatenate((x1.data, x2.data)),
            jnp.concatenate((x1.row, x2.row)),
            jnp.concatenate((x1.col, x2.col)),
        ),
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
    return COO((data, mat.row, mat.col), shape=mat.shape)


def mul(mat: COO, other: tp.Union[int, float, jnp.ndarray]) -> COO:
    """Element-wise product, potentially with broadcasting."""
    other = jnp.asarray(other)
    if other.ndim == 0:
        return with_data(mat, mat.data * other)
    if other.ndim == 1:
        return with_data(mat, mat.data * other[mat.col])
    if other.ndim == 2:
        return with_data(mat, mat.data * other[mat.row, mat.col])
    raise NotImplementedError(
        f"other.ndim must be <= 2 but other has shape {other.shape}"
    )


def standardize(mat: COO) -> COO:
    """
    Reduce to standard form by sorting indices and summing duplicates.

    Not particularly performant. Not jitable. If indices are known to be unique, use
    `reorder` instead.
    """
    indices = indices_1d(mat.row, mat.col, mat.shape)
    indices, inverse = jnp.unique(indices, return_inverse=True)
    data = jax.ops.segment_sum(mat.data, inverse, indices_are_sorted=False)
    valid = data != 0
    data = data[valid]
    indices = indices[valid]
    row, col = jnp.unravel_index(indices, mat.shape)
    return COO((data, row, col), shape=mat.shape)


def symmetrize(mat: COO) -> COO:
    """Get `(coo + coo.T) / 2`."""
    assert len(mat.shape) == 2 and mat.shape[-2] == mat.shape[-1], mat.shape
    mat = add_coo(mat, mat.T)
    return with_data(mat, mat.data / 2)


def symmetrize_data(mat: COO) -> COO:
    """
    Get `(mat + mat.T) / 2` assuming mat has symmetric sparsity.

    Symmetric sparsity pattern is not checked.
    """
    assert len(mat.shape) == 2 and mat.shape[-2] == mat.shape[-1], mat.shape
    mat_t = mat.T
    perm = reorder_perm(mat_t.row, mat_t.col, mat_t.shape)
    return with_data(mat, (mat.data + mat.data[perm]) / 2)


def masked_inner(mat: COO, x, y) -> jnp.ndarray:
    """Comput `(x.T @ y)[row, col]`."""
    assert x.ndim == 2, x.shape
    assert y.ndim == 2, y.shape
    return (x[:, mat.row] * y[:, mat.col]).sum(axis=0)


def masked_outer(mat: COO, x, y) -> jnp.ndarray:
    """Compute `(x @ y.T)[row, col]`."""
    if x.ndim == 1:
        assert y.ndim == 1, (x.shape, y.shape)
        return x[mat.row] * y[mat.col]
    elif x.ndim == 2:
        assert y.ndim == 2, (x.shape, y.shape)
        assert (x.shape[0], y.shape[0]) == mat.shape and x.shape[1] == y.shape[1], (
            x.shape,
            y.shape,
            mat.shape,
        )
        return (x[mat.row] * y[mat.col]).sum(axis=1)
    raise ValueError(f"x and y must each be rank 1 or 2, got {x.shape}")


def masked_data(mat: COO, x: jnp.ndarray) -> jnp.ndarray:
    assert x.ndim == 2, x.shape
    return x[mat.row, mat.col]


def scale_rows(mat: COO, x: jnp.ndarray) -> COO:
    return with_data(mat, mat.data * x[mat.row])


def scale_columns(mat: COO, x: jnp.ndarray) -> COO:
    return with_data(mat, mat.data * x[mat.col])


def negate(mat: COO) -> COO:
    return with_data(mat, -mat.data)


def _reduce(mat: COO, axis, segment_reduction: tp.Callable) -> jnp.ndarray:
    if not isinstance(axis, int):
        raise NotImplementedError("TODO")
    axis = canonicalize_axis(axis, 2)
    if axis == 0:
        indices = mat.col
        num_segments = mat.shape[1]
    elif axis == 1:
        indices = mat.row
        num_segments = mat.shape[0]
    else:
        raise NotImplementedError("TODO")
    return segment_reduction(mat.data, indices, num_segments=num_segments)


def sum(mat: COO, axis=None) -> jnp.ndarray:  # pylint: disable=redefined-builtin
    if axis is None:
        return mat.data.sum()
    return _reduce(mat, axis, jax.ops.segment_sum)


def max(mat: COO, axis=None) -> jnp.ndarray:  # pylint: disable=redefined-builtin
    if axis is None:
        return mat.data.max()
    return _reduce(mat, axis, partial(segment_max, initial=0))


def _boolean_mask(
    mat: COO, mask: jnp.ndarray, valid_indices: jnp.ndarray, axis: int
) -> COO:
    assert jnp.issubdtype(mask.dtype, jnp.bool_), mask.dtype
    assert axis in (0, 1), axis
    coords = mat.row, mat.col
    valid = mask[coords[axis]]
    remapped = (
        jnp.zeros((mat.shape[axis],), dtype=mat.row.dtype)
        .at[valid_indices]
        .set(jnp.arange(valid_indices.size))
    )
    coords = [mat.row[valid], mat.col[valid]]
    coords[axis] = remapped[coords[axis]]
    data = mat.data[valid]
    shape = list(mat.shape)
    shape[axis] = valid_indices.size
    return COO((data, *coords), shape=tuple(shape))


def boolean_mask(mat: COO, mask: jnp.ndarray, axis: int = 0):
    mask = jnp.asarray(mask, dtype=bool)
    assert mask.ndim == 1, mask.shape
    axis = canonicalize_axis(axis, 2)
    (valid_indices,) = jnp.where(mask)
    return _boolean_mask(mat, mask, valid_indices, axis)


def gather(mat: COO, indices: jnp.ndarray, axis: int = 0):
    axis = canonicalize_axis(axis, 2)
    mask = jnp.zeros((mat.shape[axis],), bool).at[indices].set(True)
    return _boolean_mask(mat, mask, indices, axis)


def get_coords(mat: COO, axis: int):
    axis = canonicalize_axis(axis, 2)
    assert axis in (0, 1)
    return mat.row if axis == 0 else mat.col


def softmax(mat: COO, axis: int = -1) -> COO:
    return with_data(
        mat, segment_softmax(mat.data, get_coords(mat, axis), mat.shape[axis])
    )


def to_coo(coo: COO) -> COO:
    return coo


def to_csr(coo: COO) -> CSR:
    nrows = coo.shape[0]
    indptr = jnp.zeros(nrows + 1, coo.row.dtype)
    indptr = indptr.at[1:].set(jnp.cumsum(jnp.bincount(coo.row, length=nrows)))
    return CSR((coo.data, coo.col, indptr), shape=coo.shape)


def to_dense(coo: COO) -> jnp.ndarray:
    # use add instead of set in case coo has duplicates
    return jnp.zeros(coo.shape, coo.dtype).at[coo.row, coo.col].add(coo.data)


def transpose(coo: COO) -> jnp.ndarray:
    return reorder(COO((coo.data, coo.col, coo.row), shape=coo.shape[-1::-1]))
