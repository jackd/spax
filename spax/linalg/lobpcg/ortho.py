from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from spax.linalg.lobpcg import utils
from spax.linalg.utils import as_array_fun, standardize_signs
from spax.types import ArrayFun, ArrayOrFun


class OrthoState(NamedTuple):
    iterations: int
    theta: jnp.ndarray
    X: jnp.ndarray
    P: jnp.ndarray
    R: jnp.ndarray
    err: jnp.ndarray
    converged: jnp.ndarray
    num_converged: int


def lobpcg(
    A: ArrayOrFun,
    X0: jnp.ndarray,
    B: Optional[ArrayOrFun] = None,
    # iK: Optional[ArrayOrFun] = None,
    largest: bool = False,
    k: Optional[int] = None,
    tol: Optional[float] = None,
    max_iters: int = 1000,
    tau_ortho: Optional[float] = None,
    tau_replace: Optional[float] = None,
    tau_drop: Optional[float] = None,
    # tau_skip: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, OrthoState]:
    """
    Find some of the eigenpairs for the generalized eigenvalue problem (A, B).

    Args:
        A: `[m, m]` hermitian matrix, or function representing pre-multiplication by an
            `[m, m]` hermitian matrix.
        X0: `[m, n]`, `k <= n < m`. Initial guess of eigenvectors.
        B: same type as A. If not given, identity is used.
        iK: Optional inverse preconditioner. If not given, identity is used.
        largest: if True, return the largest `k` eigenvalues, otherwise the smallest.
        k: number of eigenpairs to return. Uses `n` if not provided.
        tol: tolerance for convergence.
        max_iters: maximum number of iterations.
        tau_*: solver parameters.

    Returns:
        w: [k] smallest/largest eigenvalues of generalized eigenvalue problem `(A, B)`.
        v: [n, k] eigenvectors associated with `w`. `v[:, i]` matches `w[i]`.
        iters: number of iterations used.
    """
    # Perform argument checks and fix default / computed arguments
    if B is not None:
        raise NotImplementedError("Implementations with non-None B have issues")
    # if iK is not None:
    #     raise NotImplementedError("Inplementations with non-None iK have issues")
    ohm = jax.random.normal(jax.random.PRNGKey(0), shape=X0.shape, dtype=X0.dtype)
    A = as_array_fun(A)
    A_norm = utils.approx_matrix_norm2(A, ohm)
    if B is None:
        B = utils.identity
        B_norm = jnp.ones((), dtype=X0.dtype)
    else:
        B = as_array_fun(B)
        B_norm = utils.approx_matrix_norm2(B, ohm)
    # if iK is None:
    #     iK = utils.identity
    # else:
    #     iK = as_array_fun(iK)

    if tol is None:
        dtype = X0.dtype
        if dtype == jnp.float32:
            feps = 1.2e-7
        elif dtype == jnp.float64:
            feps = 2.23e-16
        else:
            raise KeyError(dtype)
        tol = feps ** 0.5

    k = k or X0.shape[1]
    return _lobpcg(
        A,
        X0,
        B,
        # iK,
        largest,
        k,
        tol,
        max_iters,
        A_norm,
        B_norm,
        tau_ortho=tau_ortho or tol,
        tau_replace=tau_replace or tol,
        tau_drop=tau_drop or tol,
        # tau_skip=tau_skip or tol,
    )


def _lobpcg(
    A: ArrayFun,
    X0: jnp.ndarray,
    B: ArrayFun,
    # iK: ArrayFun,
    largest: bool,
    k: int,
    tol: float,
    max_iters: int,
    A_norm: float,
    B_norm: float,
    tau_ortho: float,
    tau_replace: float,
    tau_drop: float,
    # tau_skip: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, OrthoState]:
    m, nx = X0.shape
    dtype = X0.dtype
    compute_residual = partial(utils.compute_residual, A=A, B=B)
    compute_residual_error = partial(
        utils.compute_residual_error, A_norm=A_norm, B_norm=B_norm
    )
    ortho_drop = partial(
        utils.ortho_drop,
        B,
        tau_replace=tau_replace,
        tau_drop=tau_drop,
        largest=largest,
        tau_ortho=tau_ortho,
    )

    theta, C = utils.rayleigh_ritz(X0, A, B, largest=largest)
    X = X0 @ C
    del X0
    P = jnp.zeros((m, 0), dtype=dtype)
    R = compute_residual(theta, X)
    err = compute_residual_error(R, theta, X)
    converged = err < tol
    num_converged = jnp.count_nonzero(converged)

    state = OrthoState(
        iterations=0,
        theta=theta,
        X=X,
        P=P,
        R=R,
        err=err,
        converged=converged,
        num_converged=num_converged,
    )
    del theta

    def cond(s: OrthoState):
        return jnp.logical_and(s.iterations < max_iters, s.num_converged < k)

    def body(s: OrthoState):
        XP = jnp.concatenate((s.X, s.P), axis=1)
        W = ortho_drop(U=s.R, V=XP)
        S = jnp.concatenate((XP, W), axis=1)
        theta_x, theta_p, cx, cp = utils.rayleigh_ritz_modified_ortho(
            S=S, A=A, nx=nx, nc=s.num_converged, largest=largest
        )
        del theta_p
        X = S @ cx
        P = S @ cp
        R = compute_residual(theta_x, X)
        err = compute_residual_error(R, theta_x, X)
        converged = err < tol
        num_converged = jnp.count_nonzero(converged)
        return OrthoState(
            iterations=s.iterations + 1,
            theta=theta_x,
            X=X,
            P=P,
            R=R,
            err=err,
            converged=converged,
            num_converged=num_converged,
        )

    # main loop
    while cond(state):
        state = body(state)

    # # first run through has P=[]. Size will change for subsequent runs.
    # state = body(state)
    # state = jax.lax.while_loop(cond, body, state)

    # clean up return values
    def if_converged(state):
        indices = jnp.argsort(jnp.logical_not(state.converged))[:k]
        theta = state.theta[indices]
        vectors = state.X[:, indices]
        vectors = standardize_signs(vectors)
        return theta, vectors, state

    def otherwise(state):
        theta = jnp.full((k,), jnp.nan, dtype=dtype)
        vectors = jnp.full((m, k), jnp.nan, dtype=dtype)
        return theta, vectors, state

    pred = state.num_converged >= k
    return jax.lax.cond(pred, if_converged, otherwise, state)

    # use_ortho = False
    # while nc < k and iters < max_iters:
    #     iters += 1
    #     if use_ortho:
    #         W = utils.ortho_drop(
    #             B, R, XP, tau_ortho, tau_replace, tau_drop, largest=largest
    #         )
    #     else:
    #         W = R
    #     S = jnp.concatenate((XP, W), axis=1)
    #     theta, C, next_use_ortho = utils.rayleigh_ritz_modified(
    #         S, A, B, k, nx, nc, use_ortho, tau_skip, largest=largest
    #     )
    #     if use_ortho != next_use_ortho:
    #         W = utils.ortho_drop(
    #             B, R, XP, tau_ortho, tau_replace, tau_drop, largest=largest
    #         )
    #         # The line below isn't in the pseudocode, but otherwise W isn't used.
    #         S = jnp.concatenate((XP, W), axis=1)
    #         theta, C, use_ortho = utils.rayleigh_ritz_modified(
    #             S, A, B, k, nx, nc, use_ortho, tau_skip, largest=largest
    #         )
    #     XP = S @ C
    #     X = X[:, :nx]
    #     R = compute_residual(theta, X)
    #     rerr = compute_residual_error(R, theta, X)
    #     nc = jnp.count_nonzero(rerr < tol)
