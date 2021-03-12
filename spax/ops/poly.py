import typing as tp
from functools import partial

import jax.numpy as jnp

from spax import utils
from spax.ops import bsr, coo, csr, dense, ell
from spax.sparse import COO, SparseArray

S = tp.TypeVar("S", bound=tp.Union[SparseArray, jnp.ndarray])


def _get_lib(mat: SparseArray):
    if utils.is_coo(mat):
        return coo
    if utils.is_bsr(mat):
        return bsr
    if utils.is_csr(mat):
        return csr
    if utils.is_ell(mat):
        return ell
    if utils.is_dense(mat):
        return dense
    raise TypeError(f"Unrecognized `SparseArray` type {type(mat)}")


def _delegate(mat: SparseArray, method_name: str, *args, **kwargs):
    lib = _get_lib(mat)
    method = getattr(lib, method_name, None)
    if method is None:
        raise NotImplementedError(
            f"{method_name} not implemented for format {type(mat)}"
        )
    return method(*args, **kwargs)


def matmul(mat: SparseArray, v) -> jnp.ndarray:
    return _delegate(mat, "matmul", mat, v)


def transpose(mat: S, axes=None) -> S:
    return _delegate(mat, "transpose", mat, axes=axes)


def add(mat: S, other) -> tp.Union[S, jnp.ndarray]:
    return _delegate(mat, "add", mat, other)


def mul(mat: S, other) -> S:
    return _delegate(mat, "mul", mat, other)


def symmetrize(mat: S) -> S:
    return _delegate(mat, "symmetrize", mat)


def masked_inner(mat: S, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return _delegate(mat, "masked_inner", mat, x, y)


def masked_outer(mat: S, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return _delegate(mat, "masked_outer", mat, x, y)


def masked_matmul(mat: S, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return masked_inner(mat, x.T, y)


def with_data(mat: S, data: jnp.ndarray) -> S:
    return _delegate(mat, "with_data", mat, data)


def map_data(mat: S, fun: tp.Callable[[jnp.ndarray], jnp.ndarray]) -> S:
    if hasattr(mat, "data"):
        return with_data(mat, fun(mat.data))
    mask = mat != 0
    return mat.at[mask].set(fun(mat[mask]))


def conj(mat: S) -> S:
    return map_data(mat, jnp.conj)


def abs(mat: S) -> S:  # pylint: disable=redefined-builtin
    return map_data(mat, jnp.abs)


def remainder(x1: S, x2) -> S:
    return map_data(x1, partial(jnp.remainder, x2=x2))


def to_dense(mat: tp.Union[SparseArray, jnp.ndarray]) -> jnp.ndarray:
    if isinstance(mat, SparseArray):
        return mat.todense()
    return jnp.asarray(mat)


def scale(mat: S, value: float) -> S:
    return map_data(mat, lambda d: d * value)


def scale_rows(mat: S, x: jnp.ndarray) -> S:
    return _delegate(mat, "scale_rows", mat, x)


def scale_columns(mat: S, x: jnp.ndarray) -> S:
    return _delegate(mat, "scale_columns", mat, x)


def negate(mat: S) -> S:
    return _delegate(mat, "negate", mat)


def subtract(mat: S, other) -> S:
    return _delegate(mat, "subtract", mat, other)


def sum(mat: S, axis=None) -> jnp.ndarray:
    return _delegate(mat, "sum", mat, axis)


def max(mat: S, axis=None) -> jnp.ndarray:
    return _delegate(mat, "max", mat, axis)


def boolean_mask(mat: S, mask: jnp.ndarray, axis: int = 0) -> S:
    return _delegate(mat, "boolean_mask", mat, mask, axis)


def gather(mat: S, indices: jnp.ndarray, axis: int = 0) -> S:
    return _delegate(mat, "gather", mat, indices, axis)


def symmetrize_data(mat: S) -> S:
    return _delegate(mat, "symmetrize_data", mat)


def norm(
    mat: S, ord: tp.Union[int, str] = 2, axis: tp.Optional[int] = None
) -> jnp.ndarray:
    if axis is None:
        raise NotImplementedError("`axis` must be provided")
    if ord == 2:
        return jnp.sqrt(sum(map_data(mat, lambda d: d * d.conj()), axis=axis))
    if ord == 1:
        return sum(abs(mat), axis=axis)
    if ord == ord == jnp.inf:
        return max(abs(mat), axis=axis)


def to_dense(mat: S) -> jnp.ndarray:
    if hasattr(mat, "todense"):
        return mat.todense()
    assert isinstance(mat, jnp.ndarray), type(mat)
    return mat


def cast(mat: S, dtype: jnp.ndarray):
    if mat.dtype is dtype:
        return mat
    if utils.is_sparse(mat):
        return with_data(mat, mat.data.astype(dtype))
    return mat.astype(dtype)


def softmax(mat: S, axis=-1):
    return _delegate(mat, "softmax", mat, axis)


def to_coo(mat: S) -> COO:
    return _delegate(mat, "to_coo", mat)


def get_coords(mat: S) -> jnp.ndarray:
    return _delegate(mat, "get_coords", mat)
