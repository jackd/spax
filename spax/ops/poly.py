import typing as tp
from functools import partial

import jax.numpy as jnp
from jax.experimental.sparse.ops import COO, CSR, JAXSparse

from spax.ops import coo, csr, dense

S = tp.TypeVar("S", bound=tp.Union[JAXSparse, jnp.ndarray])


def _get_lib(mat: JAXSparse):
    if isinstance(mat, COO):
        return coo
    if isinstance(mat, CSR):
        return csr
    if isinstance(mat, jnp.ndarray):
        return dense
    raise TypeError(f"Unsupported `JAXSparse`: {type(mat)}")


def _delegate(mat: JAXSparse, method_name: str, *args, **kwargs):
    lib = _get_lib(mat)
    method = getattr(lib, method_name, None)
    if method is None:
        raise NotImplementedError(
            f"{method_name} not implemented for format {type(mat)}"
        )
    return method(*args, **kwargs)


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


def scale(mat: S, value: float) -> S:
    return map_data(mat, lambda d: d * value)


def scale_rows(mat: S, x: jnp.ndarray) -> S:
    return _delegate(mat, "scale_rows", mat, x)


def scale_columns(mat: S, x: jnp.ndarray) -> S:
    return _delegate(mat, "scale_columns", mat, x)


def negate(mat: S) -> S:
    return _delegate(mat, "negate", mat)


def sum(mat: S, axis=None) -> jnp.ndarray:  # pylint: disable=redefined-builtin
    return _delegate(mat, "sum", mat, axis)


def max(mat: S, axis=None) -> jnp.ndarray:  # pylint: disable=redefined-builtin
    return _delegate(mat, "max", mat, axis)


def boolean_mask(mat: S, mask: jnp.ndarray, axis: int = 0) -> S:
    return _delegate(mat, "boolean_mask", mat, mask, axis)


def gather(mat: S, indices: jnp.ndarray, axis: int = 0) -> S:
    return _delegate(mat, "gather", mat, indices, axis)


def symmetrize_data(mat: S) -> S:
    return _delegate(mat, "symmetrize_data", mat)


def norm(
    mat: S,
    ord: tp.Union[int, str] = 2,  # pylint: disable=redefined-builtin
    axis: tp.Optional[int] = None,
) -> jnp.ndarray:
    if axis is None:
        raise NotImplementedError("`axis` must be provided")
    if ord == 2:
        return jnp.sqrt(sum(map_data(mat, lambda d: d * d.conj()), axis=axis))
    if ord == 1:
        return sum(abs(mat), axis=axis)
    if ord == jnp.inf:
        return max(abs(mat), axis=axis)
    raise NotImplementedError(f"ord {ord} not implemented")


def cast(mat: S, dtype: jnp.ndarray):
    if mat.dtype is dtype:
        return mat
    if isinstance(mat, JAXSparse):
        return with_data(mat, mat.data.astype(dtype))
    return mat.astype(dtype)


def softmax(mat: S, axis=-1):
    return _delegate(mat, "softmax", mat, axis)


def to_coo(mat: S) -> COO:
    return _delegate(mat, "to_coo", mat)


def to_csr(mat: S) -> CSR:
    return _delegate(mat, "to_csr", mat)


def to_dense(mat: S) -> jnp.ndarray:
    if isinstance(mat, JAXSparse):
        return coo.to_dense(to_coo(mat))
    if isinstance(mat, jnp.ndarray):
        return mat
    raise TypeError(f"Invalid mat type {type(mat)}")


def get_coords(mat: S) -> jnp.ndarray:
    return _delegate(mat, "get_coords", mat)


def add_dense(mat: S, other: tp.Union[int, float, jnp.ndarray]):
    other = jnp.asarray(other)
    if other.ndim < 2:
        other = jnp.broadcast_to(other, mat.shape)
    mat = to_coo(mat)
    return other.at[..., mat.row, mat.col].add(mat.data)


def add(
    mat: S, other: tp.Union[JAXSparse, int, float, jnp.ndarray]
) -> tp.Union[COO, jnp.ndarray]:
    if isinstance(mat, JAXSparse):
        if isinstance(other, JAXSparse):
            return coo.add_coo(to_coo(mat), to_coo(other))
        return add_dense(mat, other)
    assert isinstance(mat, jnp.ndarray)
    if isinstance(other, JAXSparse):
        return add_dense(to_coo(other), mat)
    return mat + other


def subtract(
    mat: tp.Union[jnp.ndarray], other: tp.Union[JAXSparse, int, float, jnp.ndarray]
):
    if isinstance(other, JAXSparse):
        other = negate(other)
    else:
        other = -other
    return add(mat, other)
