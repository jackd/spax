import typing as tp
from functools import partial

import jax
import jax.numpy as jnp

from spax.linalg import polynomials as poly
from spax.linalg import utils
from spax.types import ArrayFun, ArrayOrFun


def eigh(a, largest=True):
    w, v = jnp.linalg.eigh(a)
    return _sort_by_magnitude(w, v, largest=largest)


def _sort_by_magnitude(w, v, largest=True):
    if largest:
        perm = jnp.argsort(-jnp.abs(w))
    else:
        perm = jnp.argsort(jnp.abs(w))

    w = w[perm]
    v = v[:, perm]
    return w, v


class SubspaceIterationInfo(tp.NamedTuple):
    err: jnp.ndarray
    iterations: int


def default_tol(dtype: jnp.dtype) -> float:
    return jnp.finfo(dtype).eps ** 0.5


def no_accelerator(v, av) -> jnp.ndarray:
    del v
    return av


def _scaled(a, scale, x):
    return a(x) / scale


def _chebyshev_accelerator(
    order: int, scale: float, a: ArrayFun, v: jnp.ndarray, av: jnp.ndarray
) -> jnp.ndarray:
    order = 8  # HACK
    a_scaled = partial(_scaled, a, jnp.asarray(scale, v.dtype))
    for _ in range(order - 2):
        v, av = poly.iterate_chebyshev1(a_scaled, v, av)
    return av
    # with loops.Scope() as scope:
    #     scope.v = v
    #     scope.av = av
    #     for _ in scope.range(order):
    #         scope.v, scope.av = poly.iterate_chebyshev1(a_scaled, scope.v, scope.av)
    #     av = scope.av
    # return av


def chebyshev_accelerator_fun(order: int, scale: float, a: ArrayOrFun):
    # assert order >= 2
    return jax.tree_util.Partial(
        _chebyshev_accelerator, order, scale, utils.as_array_fun(a)
    )


def basic_subspace_iteration(
    a: ArrayOrFun,
    v0: jnp.ndarray,
    tol: tp.Optional[float] = None,
    maxiters: int = 1000,
    accelerator: tp.Callable = no_accelerator,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray, SubspaceIterationInfo]:
    """
    See http://www.netlib.org/utk/people/JackDongarra/etemplates/node98.html

    Args:
        accelerator: Function mapping (v, av) -> z.
        a: [m, m] matrix or matmul function to find the dominant eigenvpairs of.
        v0: [m, k] initial estimate of eigenvectors.
        tol: tolerance, defaults to eps ** 0.5 if not given.
        maxiters: maximum number of iterations.

    Returns:
        w: [k] eigenvalues.
        v: [m, k] eivenvectors.
        info: `SubspaceIterationInfo`, `tp.NamedTuple` with
            err: [k] error
            iterations: int
    """
    if tol is None:
        tol = default_tol(v0.dtype)

    a = utils.as_array_fun(a)

    def cond(state):
        w, v, av, err, iters = state
        del w, v, av
        return jnp.logical_and(jnp.max(err) > tol, iters < maxiters)

    def body(state):
        w, v, av, err, iters = state
        del w, err
        z = accelerator(v, av)
        v, _ = jnp.linalg.qr(z)
        av = a(v)
        w = utils.rayleigh_quotient(v, av)
        w, v = _sort_by_magnitude(w, v)
        err = jnp.linalg.norm(av - v * w, axis=0)
        return w, v, av, err, iters + 1

    state = (None, v0, a(v0), None, 0)
    state = body(state)
    w, v, av, err, iters = jax.lax.while_loop(cond, body, state)

    del av
    return w, v, SubspaceIterationInfo(err, iters)


# @partial(jax.jit, static_argnums=0)
def chebyshev_subspace_iteration(
    order: int,
    scale: float,
    a: ArrayOrFun,
    v0: jnp.ndarray,
    tol: tp.Optional[float] = None,
    maxiters: int = 1000,
):
    return basic_subspace_iteration(
        a,
        v0,
        tol=tol,
        maxiters=maxiters,
        accelerator=chebyshev_accelerator_fun(order, scale, a),
    )


def projected_subspace_iteration(
    a: ArrayOrFun,
    v0: jnp.ndarray,
    tol: tp.Optional[float] = None,
    maxiters: int = 1000,
    accelerator: tp.Callable = no_accelerator,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray, SubspaceIterationInfo]:
    """Algorithm 5.3 from Saad 2011."""
    if tol is None:
        tol = default_tol(v0.dtype)
    a = utils.as_array_fun(a)

    def cond(state):
        w, v, av, err, iters = state
        del w, v, av
        return jnp.logical_and(jnp.max(err) > tol, iters < maxiters)

    def body(state):
        w, v, av, err, iters = state
        del w, err
        z = accelerator(v, av)
        z, _ = jnp.linalg.qr(z)
        b = z.conj().T @ a(z)
        w, v = jnp.linalg.eigh(b)
        v = z @ v
        v = v / jnp.linalg.norm(v, axis=0)
        av = a(v)
        err = jnp.linalg.norm(av - v * w, axis=0)
        return w, v, av, err, iters + 1

    state = (None, v0, a(v0), None, 0)
    state = body(state)
    w, v, av, err, iters = jax.lax.while_loop(cond, body, state)
    del av
    # reverse order so consistent with other subspace_iteration implementations
    w, v = _sort_by_magnitude(w, v)
    return w, v, SubspaceIterationInfo(err, iters)


# @partial(jax.jit, static_argnums=0)
def chebyshev_projected_subspace_iteration(
    order: int,
    scale: float,
    a: ArrayOrFun,
    v0: jnp.ndarray,
    tol: tp.Optional[float] = None,
    maxiters: int = 1000,
):
    return projected_subspace_iteration(
        a,
        v0,
        tol=tol,
        maxiters=maxiters,
        accelerator=chebyshev_accelerator_fun(order, scale, a),
    )


def locking_projected_subspace_iteration(
    a: ArrayOrFun,
    v0: jnp.ndarray,
    tol: tp.Optional[float] = None,
    maxiters: int = 1000,
    accelerator: tp.Callable = no_accelerator,
    # locked: tp.Optional[jnp.ndarray] = None,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray, SubspaceIterationInfo]:
    if tol is None:
        tol = default_tol(v0.dtype)
    a = utils.as_array_fun(a)
    locked = jnp.zeros((v0.shape[0], 0), dtype=v0.dtype)
    # if locked is None:
    #     locked = jnp.zeros((v0.shape[0], 0), dtype=v0.dtype)
    # else:
    #     locked, _ = jnp.linalg.qr(locked)
    # assert locked.ndim == 2
    nl0 = locked.shape[1]

    def cond(state):
        w, v, av, err, iters = state
        del w, v, av
        return jnp.logical_and(iters < maxiters, err[0] > tol)

    def body(locked, state):
        nl = locked.shape[1]
        w, v, av, err, iters = state
        z = accelerator(v, av)
        if nl > 0:
            # subtract projection of z onto locked
            z = z - locked @ locked.T @ av
        # orthogonalize z with itself
        z, _ = jnp.linalg.qr(z)
        # creat full z
        z = jnp.concatenate((locked, z), axis=1)
        # apply Rayleigh-Ritz.
        b = z.conj().T @ a(z)
        w, v = eigh(b)
        w = w[nl:]
        v = v[:, nl:]
        v = z @ v
        v = v / jnp.linalg.norm(v, axis=0)  # is this necessary?
        av = a(v)
        err = jnp.linalg.norm(av - v * w, axis=0)
        return w, v, av, err, iters + 1

    # if nl0 != 0:
    #     v0 = v0 - locked @ locked.T @ v0
    v = v0 / jnp.linalg.norm(v0, axis=0)
    av = a(v)
    w = jax.vmap(lambda vi, avi: vi @ avi, (1, 1))(v, av)  # rayleigh-quotient
    err = jnp.linalg.norm(av - v * w, axis=0)
    iters = 0
    state = (w, v, av, err, iters)

    acc_err = []
    acc_w = []

    for _ in range(v0.shape[1]):
        w, v, av, err, iters = jax.lax.while_loop(cond, partial(body, locked), state)
        v_converged, v = jnp.split(v, (1,), axis=1)
        locked = jnp.concatenate((locked, v_converged), axis=1)

        err_converged, err = jnp.split(err, (1,))
        acc_err.append(err_converged)

        w_converged, w = jnp.split(w, (1,))
        acc_w.append(w_converged)

        av = av[:, 1:]

        state = (w, v, av, err, iters)

    assert v.shape[1] == 0
    w = jnp.concatenate(acc_w)
    err = jnp.concatenate(acc_err)
    v = locked[:, nl0:]  # don't return the initially locked vectors
    return w, v, SubspaceIterationInfo(err, iters)
