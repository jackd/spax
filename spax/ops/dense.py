import jax.numpy as jnp

from spax.utils import canonicalize_axis


def mul(mat: jnp.ndarray, other) -> jnp.ndarray:
    return mat * other


def symmetrize(mat: jnp.ndarray) -> jnp.ndarray:
    return (mat + mat.T) / 2


def masked_inner(mat: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return masked_data(mat, x.T @ y)


def masked_outer(mat: jnp.ndarray, x, y) -> jnp.ndarray:
    if x.ndim == 1:
        assert y.ndim == 1, (x.shape, y.shape)
        return masked_data(mat, jnp.outer(x, y))
    elif x.ndim == 2:
        assert y.ndim == 2, (x.shape, y.shape)
        assert (x.shape[0], y.shape[0]) == mat.shape and x.shape[1] == y.shape[1], (
            x.shape,
            y.shape,
            mat.shape,
        )
        return masked_data(mat, x @ y.T)
    raise ValueError(f"x and y must each be rank 1 or 2, but have shape {x.shape}")


def masked_data(mat: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return x[mat != 0]


def scale_rows(mat: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return mat * x[:, jnp.newaxis]


def scale_columns(mat: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return mat * x


def negate(mat: jnp.ndarray) -> jnp.ndarray:
    return -mat


def sum(  # pylint: disable=redefined-builtin
    mat: jnp.ndarray, axis=None
) -> jnp.ndarray:
    return mat.sum(axis)


def max(  # pylint: disable=redefined-builtin
    mat: jnp.ndarray, axis=None
) -> jnp.ndarray:
    return mat.max(axis)


def _get_on_axis(mat, op, axis):
    axis = canonicalize_axis(axis, mat.ndim)
    empty = slice(None, None, None)
    return mat.__getitem__((empty,) * axis + (op,))


def boolean_mask(mat: jnp.ndarray, mask: jnp.ndarray, axis: int = 0):
    return _get_on_axis(mat, mask, axis)


def gather(mat: jnp.ndarray, indices: jnp.ndarray, axis: int = 0):
    return _get_on_axis(mat, indices, axis)


def with_data(mat: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
    ids = jnp.where(mat != 0)
    return jnp.zeros(mat.shape, data.dtype).at[ids].set(data)
