import jax
import jax.numpy as jnp

from spax import is_sparse, ops
from spax.types import ArrayFun, ArrayOrFun


def _standardize_signs(v):
    assert v.ndim == 1
    val = v[jnp.argmax(jnp.abs(v))]
    if val.dtype in (jnp.complex64, jnp.complex128):
        return v * jnp.abs(val) / val  # make real
    return v * jnp.sign(val)


def standardize_signs(v: jnp.ndarray) -> jnp.ndarray:
    """Get `w = s*v` such that `max(abs(w)) == max(w) >= 0` and `abs(s) == 1`."""
    if v.ndim == 0:
        return jnp.abs(v)
    shape = v.shape
    v = v.reshape((shape[0], -1))
    v = jax.vmap(_standardize_signs, 1, 1)(v)
    return v.reshape(shape)


def rayleigh_quotient(v, av):
    """
    Assuming columns of v are normalized and av = a(v), get `v_i.T @ a(v_i)`.

    Args:
        v: [n, k] vectors (e.g. eigenvector approximations), assumed normalized along
            dim 0.
        av: [n, k] a(v) for some linear operator a

    Returns:
        w: [k] the rayleigh quotient of the columns of `v` w.r.t `a`.
    """
    assert v.ndim == 2
    assert av.ndim == 2
    return jax.vmap(lambda vi, avi: vi @ avi, (1, 1))(v, av)


def as_array_fun(a: ArrayOrFun) -> ArrayFun:
    if callable(a):
        return a
    assert a.ndim == 2, a.shape
    if is_sparse(a):
        return jax.tree_util.Partial(ops.matmul, a)
    if isinstance(a, jnp.ndarray):
        return jax.tree_util.Partial(jnp.matmul, a)
    raise TypeError(f"a must be a callable, jax array or SparseArray, got {a}")


def _deflate_mat(a: jnp.ndarray, s: float, u: jnp.ndarray) -> jnp.ndarray:
    """Get `a - s * u @ u.T`."""
    assert a.ndim == 2
    assert u.ndim == 2
    assert a.shape[0] == a.shape[1]
    assert a.shape[0] == u.shape[0]
    return a - s * u @ u.T


def _eval_defalted(
    a: ArrayFun, s: float, u: jnp.ndarray, x: jnp.ndarray
) -> jnp.ndarray:
    assert u.ndim == 2
    return a(x) - s * u @ (u.T @ x)


def _deflate_fun(a: ArrayFun, s: float, u: jnp.ndarray,) -> ArrayFun:
    """Get the matrix function `lambda x: a(x) - s * u @ u.T @ x`."""
    return jax.tree_util.Partial(_eval_defalted, a, s, u)


def deflate(a: ArrayOrFun, s: float, u: jnp.ndarray) -> ArrayOrFun:
    """Get the matrix of function equivalent of `(a  - s * u @ u.T)`."""
    if u.ndim == 1:
        u = u[:, jnp.newaxis]
    assert u.ndim == 2
    if isinstance(a, jnp.ndarray):
        return _deflate_mat(a, s, u)
    return _deflate_fun(a, s, u)


def _eval_deflated_eigenvector(a: ArrayFun, u: jnp.ndarray, x):
    return a(x - u @ (u.T @ x))


def deflate_eigenvector(a: ArrayOrFun, u: jnp.ndarray) -> ArrayFun:
    if u.ndim == 1:
        u = u[:, jnp.newaxis]
    assert u.ndim == 2
    return jax.tree_util.Partial(_eval_deflated_eigenvector, as_array_fun(a), u)
