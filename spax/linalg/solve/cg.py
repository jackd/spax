import typing as tp

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves

from spax.linalg.utils import as_array_fun
from spax.types import ArrayOrFun


@jax.tree_util.Partial
def _identity(x):
    return x


def vdot_real(x, y):
    result = jnp.dot(x.real, y.real)
    if jnp.iscomplexobj(x) and jnp.iscomplexobj(y):
        result += jnp.dot(x.imag, y.imag)
    return result


class SolverInfo(tp.NamedTuple):
    residual: jnp.ndarray
    iterations: int


def cg_solve(
    a,
    b,
    x0,
    *,
    tol: float = 1e-5,
    maxiter: tp.Optional[int] = None,
    atol: float = 0.0,
    M=_identity,
    reprojector=_identity,
):
    if maxiter is None:
        size = sum(bi.size for bi in tree_leaves(b))
        maxiter = 10 * size  # copied from scipy
    bs = vdot_real(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma if M is _identity else vdot_real(r, r)
        # print(rs, atol2)
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        ap = a(p)
        alpha = gamma / vdot_real(p, ap)
        x_ = x + alpha * p
        r_ = r - alpha * ap
        z_ = M(r_)
        gamma_ = vdot_real(r_, z_)
        beta_ = gamma_ / gamma
        p_ = z_ + beta_ * p
        p_ = reprojector(p_)
        return x_, r_, gamma_, p_, k + 1

    r0 = b - a(x0)
    p0 = z0 = M(r0)
    gamma0 = vdot_real(r0, z0)
    state = (x0, r0, gamma0, p0, 0)

    x_final, r, gamma, p0, iters = jax.lax.while_loop(cond_fun, body_fun, state)
    # while cond_fun(state):
    #     state = body_fun(state)

    # x_final, r, *_, iters = state
    # print("iters")
    # print(iters)
    # print("r")
    # print(gamma)
    # print(f"maxiter = {maxiter}")
    del gamma, p0
    return x_final, SolverInfo(r, iters)


def cg_least_squares(a: ArrayOrFun, b: jnp.ndarray, x0: jnp.ndarray, **kwargs):
    a = as_array_fun(a)
    ata = lambda x: a(a(x).conj()).conj()
    l0, info = cg_solve(ata, a(b.conj()).conj(), x0, **kwargs)

    return l0, info
