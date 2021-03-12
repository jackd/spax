import jax
import jax.numpy as jnp
from jax.experimental import host_callback

from spax.linalg.utils import as_array_fun
from spax.types import ArrayOrFun


def _append(v, w):
    assert v.ndim == w.ndim + 1
    assert v.shape[1:] == w.shape
    return jnp.concatenate((v, jnp.expand_dims(w, axis=0)), axis=0)


def _pop(x):
    n = x.shape[0] - 1
    v, w = jnp.split(x, [n])
    w = jnp.squeeze(w, axis=0)
    return v, w


def _block_matmul(x, w, v, a):
    v_dot, w_dot = _pop(x)
    del x
    top = w * v_dot - a(v_dot) + v * w_dot  # [wI - A, v] @ [v_dot, w_dot]
    bottom = jnp.dot(v, v_dot)  # [v, 0] @ [v_dot, w_dot]
    return _append(top, bottom)


def eigh_single_jvp(
    a: ArrayOrFun,
    a_dot: ArrayOrFun,
    w: jnp.ndarray,
    v: jnp.ndarray,
    w_dot0: jnp.ndarray,
    v_dot0: jnp.ndarray,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter=None,
):
    dtype = w.dtype
    assert w.ndim == 0, w
    assert isinstance(w_dot0, float) or w_dot0.ndim == 0, w_dot0
    assert v.ndim == 1, v.shape
    assert v_dot0.shape == v.shape
    assert a.dtype == dtype, (a.dtype, dtype)
    assert a_dot.dtype == dtype, (a_dot.dtype, dtype)
    assert v.dtype == dtype, (v.dtype, dtype)
    assert v.dtype == dtype, (w_dot0.dtype, dtype)
    assert v_dot0.dtype == dtype, (v_dot0.dtype, dtype)
    a = as_array_fun(a)
    a_dot = as_array_fun(a_dot)

    matmul = jax.tree_util.Partial(_block_matmul, w=w, v=v, a=a)
    b = _append(a_dot(v), jnp.zeros((), dtype=w.dtype))
    x_dot0 = _append(v_dot0, w_dot0)
    x_dot, info = jax.scipy.sparse.linalg.cg(
        matmul, b, x_dot0, tol=tol, atol=atol, maxiter=maxiter
    )
    host_callback.id_print(x_dot)
    del info
    v_dot, w_dot = _pop(x_dot)
    return w_dot, v_dot


def eigh_jvp(
    a, a_dot, w, v, w_dot0, v_dot0, tol: float = 1e-5, atol: float = 0.0, maxiter=None
):
    return jax.vmap(
        jax.tree_util.Partial(eigh_single_jvp, tol=tol, atol=atol, maxiter=maxiter),
        in_axes=(None, None, 0, 1, 0, 1),
        out_axes=(0, 1),
    )(a, a_dot, w, v, w_dot0, v_dot0)
