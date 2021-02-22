from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from spax.linalg.lobpcg import utils
from spax.linalg.utils import as_array_fun, standardize_signs
from spax.types import ArrayFun, ArrayOrFun


class BasicInfo(NamedTuple):
    iterations: int
    success: bool


def lobpcg(
    A: ArrayOrFun,
    X0: jnp.ndarray,
    B: Optional[ArrayOrFun] = None,
    iK: Optional[ArrayOrFun] = None,
    largest: bool = False,
    k: Optional[int] = None,
    tol: Optional[float] = None,
    max_iters: int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, BasicInfo]:
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

    Returns:
        w: [k] smallest/largest eigenvalues of generalized eigenvalue problem `(A, B)`.
        v: [n, k] eigenvectors associated with `w`. `v[:, i]` matches `w[i]`.
        info: `BasicInfo` `namedtuple` with `(iterations: int, success: bool)`.
    """
    # Perform argument checks and fix default / computed arguments
    if B is not None:
        raise NotImplementedError("Implementations with non-None B have issues")
    if iK is not None:
        raise NotImplementedError("Inplementations with non-None iK have issues")
    ohm = jax.random.normal(jax.random.PRNGKey(0), shape=X0.shape, dtype=X0.dtype)
    A = as_array_fun(A)
    A_norm = utils.approx_matrix_norm2(A, ohm)
    if B is None:
        B = utils.identity
        B_norm = jnp.ones((), dtype=X0.dtype)
    else:
        B = as_array_fun(B)
        B_norm = utils.approx_matrix_norm2(B, ohm)
    if iK is None:
        iK = utils.identity
    else:
        iK = as_array_fun(iK)

    if tol is None:
        tol = utils.default_tol(X0.dtype)

    k = k or X0.shape[1]
    return _lobpcg(A, X0, B, iK, largest, k, tol, max_iters, A_norm, B_norm)


class _BasicState(NamedTuple):
    iteration: int
    eig_vals: jnp.ndarray  # [n]
    X: jnp.ndarray  # [m, n]
    R: jnp.ndarray  # [m, n]
    P: jnp.ndarray  # [m, n]
    failed: bool


def _lobpcg(
    A: ArrayFun,
    X0: jnp.ndarray,
    B: ArrayFun,
    iK: ArrayFun,
    largest: bool,
    k: int,
    tol: float,
    max_iters: int,
    A_norm: float,
    B_norm: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, BasicInfo]:
    m, nx = X0.shape
    dtype = X0.dtype
    rayleigh_ritz = partial(utils.rayleigh_ritz, A=A, B=B, largest=largest)
    compute_residual = partial(utils.compute_residual, A=A, B=B)
    compute_residual_error = partial(
        utils.compute_residual_error, A_norm=A_norm, B_norm=B_norm
    )

    def cond_fun(s: _BasicState):
        rerr = compute_residual_error(s.R, s.eig_vals, s.X)
        num_converged = jnp.count_nonzero(rerr < tol)
        return jnp.logical_and(
            jnp.logical_and(s.iteration < max_iters, num_converged < k),
            jnp.logical_not(s.failed),
        )

    def body_fun(s: _BasicState):
        iteration, eig_vals, X, R, P, failed = s
        S = jnp.concatenate((X, iK(R), P), axis=1)
        eig_vals, C = rayleigh_ritz(S)
        failed = jnp.logical_or(failed, jnp.any(jnp.isnan(eig_vals)))
        eig_vals = eig_vals[:nx]
        C = C[:, :nx]
        X = S @ C
        R = compute_residual(eig_vals, X)
        P = S[:, nx:] @ C[nx:]
        next_state = _BasicState(iteration + 1, eig_vals, X, R, P, failed)

        # if we failed (nans in ralyeigh_ritz) return the old state
        # with new `P` (to allow jitting - shape could be different) and `failed`

        def if_failed(operand):
            old_state, new_state = operand
            return _BasicState(*old_state[:-2], new_state.P, failed=True)

        def if_not_failed(operand):
            return operand[1]

        return jax.lax.cond(failed, if_failed, if_not_failed, (state, next_state))

    eig_vals, C = rayleigh_ritz(X0)
    X = X0 @ C
    R = compute_residual(eig_vals, X)
    P = jnp.zeros((m, 0), dtype=dtype)
    state = _BasicState(1, eig_vals, X, R, P, False)
    # unroll first loop because P changes size
    state = body_fun(state)
    state = jax.lax.while_loop(cond_fun, body_fun, state)
    w = state.eig_vals[:k]
    v = state.X[:, :k]
    v = standardize_signs(v)
    return (
        w,
        v,
        BasicInfo(
            state.iteration,
            jnp.logical_and(jnp.logical_not(state.failed), state.iteration < max_iters),
        ),
    )
