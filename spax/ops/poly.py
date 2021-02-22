import typing as tp
from functools import partial

import jax.numpy as jnp

from spax.ops import bsr, coo, csr, ell
from spax.sparse import BSR, COO, CSR, ELL, SparseArray

S = tp.TypeVar("S", bound=SparseArray)


def _get_lib(mat: SparseArray):
    if isinstance(mat, BSR):
        return bsr
    if isinstance(mat, COO):
        return coo
    if isinstance(mat, CSR):
        return csr
    if isinstance(mat, ELL):
        return ell
    raise TypeError(f"Unrecognized SparseArray type {type(mat)}")


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
    return with_data(mat, fun(mat.data))


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


def boolean_mask(mat: S, mask: jnp.ndarray, axis: int = 0) -> S:
    return _delegate(mat, "boolean_mask", mat, mask, axis)


def gather(mat: S, indices: jnp.ndarray, axis: int = 0) -> S:
    return _delegate(mat, "gather", mat, indices, axis)
