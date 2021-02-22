import typing as tp

import jax
import jax.numpy as jnp

from spax.linalg.utils import as_array_fun
from spax.types import ArrayOrFun


# decorator allows this to be passed as an argument to jitted functions
@jax.tree_util.Partial
def identity(x):
    return x


def default_tol(dtype: jnp.dtype) -> float:
    return jnp.finfo(dtype).eps ** 0.5


@jax.jit
def compute_residual(
    E: jnp.ndarray, X: jnp.ndarray, A: ArrayOrFun, B: tp.Optional[ArrayOrFun]
):
    BX = X if B is None else as_array_fun(B)(X)
    return as_array_fun(A)(X) - BX * E


@jax.jit
def approx_matrix_norm2(A: tp.Optional[ArrayOrFun], ohm: jnp.ndarray):
    """
    Approximation of matrix 2-norm of `A`.

        |A ohm|_fro / |ohm|_fro <= |A|_2

    This function returns the lower bound (left hand side).

    Args:
        A: matrix or callable that simulates matrix multiplication.
        ohm: block-vector used in formula. Should be Gaussian.

    Returns:
        Scalar, lower bound on 2-norm of A.
    """
    A = as_array_fun(A)
    return frobenius_norm(A(ohm)) / frobenius_norm(ohm)


@jax.jit
def compute_residual_error(
    R: jnp.ndarray,
    E: jnp.ndarray,
    X: jnp.ndarray,
    A_norm: float,
    B_norm: float,
    eps: float = 1e-8,
):
    R_norm = jnp.linalg.norm(R, 2, (0,))
    X_norm = jnp.linalg.norm(X, 2, (0,))
    err = jnp.where(
        R_norm < eps, jnp.zeros_like(R_norm), R_norm / (X_norm * (A_norm + E * B_norm))
    )
    # print(jnp.stack([R_norm, X_norm, err]))
    return err


@jax.jit
def eigh(a, largest: bool = False):
    """
    Get eigenvalues / eigenvectors of hermitian matrix a.

    Args:
        a: square hermitian float matrix
        largest: if True, return order is based on descending eigenvalues, otherwise
            ascending.

    Returns:
        w: [m] eigenvalues
        v: [m, m] eigenvectors
    """
    w, v = jnp.linalg.eigh(a)

    def if_true(operand):
        w, v = operand
        w = w[-1::-1]
        v = v[:, -1::-1]
        return w, v

    def if_false(operand):
        return operand

    return jax.lax.cond(largest, if_true, if_false, (w, v))


def _rayleigh_ritz_factorize(
    S: jnp.ndarray, A: ArrayOrFun, B: tp.Optional[ArrayOrFun], largest: bool = False
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    A = as_array_fun(A)
    if B is None:
        BS = S
    else:
        BS = as_array_fun(B)(S)
    SBS = S.T @ BS
    d = jnp.diag(SBS) ** -0.5  # d_right * X == X @ D
    R_lower = jnp.linalg.cholesky(SBS * d * d[:, jnp.newaxis])  # lower triangular

    # R_inv = jnp.linalg.inv(R_up)
    # RDSASDR = R_inv.T @ (d_left * (S.T @ A(S)) * d_right) @ R_inv

    DSASD = (S.T @ A(S)) * d * d[:, jnp.newaxis]
    RDSASD = jax.scipy.linalg.solve_triangular(R_lower, DSASD, lower=True)
    RDSASDR = jax.scipy.linalg.solve_triangular(R_lower, RDSASD.T, lower=True).T

    eig_vals, eig_vecs = eigh(RDSASDR, largest=largest)
    if B is not None:
        eig_vecs /= jnp.linalg.norm(eig_vecs, ord=2, axis=0)

    return eig_vals, eig_vecs, R_lower, d


def rayleigh_ritz(
    S: jnp.ndarray,
    A: ArrayOrFun,
    B: tp.Optional[ArrayOrFun] = None,
    largest: bool = False,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    """

    Based on algorithm2 of [duersch2018](
        https://epubs.siam.org/doi/abs/10.1137/17M1129830)

    Args:
        S: [m, ns] float array, matrix basis for search space. Columns must be linearly
            independent and well-conditioned with respect to `B`.
        A: Callable simulating [m, m] float matrix multiplication.
        B: Callable simulating [m, m] float matrix multiplication.

    Returns:
        (eig_vals, C) satisfying the following:
            C.T @ S.T @ B(S) @ C = jnp.eye(ns)
            C.T @ S.T @ A(S) @ C = jnp.diag(eig_vals)

        eig_vals: [ns] eigenvalues. Sorted in descending order if largest, otherwise
            ascending.
        C: [ns, ns] float matrix satisfying:
    """
    eig_vals, Z, R, d = _rayleigh_ritz_factorize(S, A, B, largest=largest)
    C = d[:, jnp.newaxis] * (jax.scipy.linalg.solve_triangular(R.T, Z, lower=False))
    return eig_vals, C


def svqb(
    B: ArrayOrFun, U: jnp.ndarray, tau_replace: float, largest: bool = False,
) -> jnp.ndarray:
    """
    Orthonormalization algorithm using SVD proposed by Stathopolous and Wu.

    Algorithm 4 from duersch2018.
    """
    UBU = U.T @ as_array_fun(B)(U)
    d = jnp.diag(UBU) ** -0.5
    DUVUD = UBU * d * d[:, jnp.newaxis]
    theta, Z = eigh(DUVUD, largest=largest)
    theta = jnp.maximum(theta, tau_replace * jnp.max(jnp.abs(theta)))
    return U @ (Z * d[:, jnp.newaxis] * (theta ** -0.5))


def svqb_drop(
    B: ArrayOrFun, U: jnp.ndarray, tau_drop: float, largest: bool = False
) -> jnp.ndarray:
    """Algorithm 6 from duersch2018."""
    UBU = U.T @ as_array_fun(B)(U)
    d = jnp.diag(UBU) ** -0.5
    DUBUD = (UBU * d * d[:, jnp.newaxis],)
    theta, Z = eigh(DUBUD, largest=largest)
    to_keep = theta > tau_drop * jnp.max(jnp.abs(theta))
    theta = theta[to_keep]
    Z = Z[:, to_keep]
    return U @ (Z * d[:, jnp.newaxis] * (theta ** -0.5))


def frobenius_norm(X: jnp.ndarray) -> float:
    return jnp.linalg.norm(X.flatten(), ord=2)


def ortho_drop(
    B: ArrayOrFun,
    U: jnp.ndarray,
    V: jnp.ndarray,
    tau_ortho: float,
    tau_replace: float,
    tau_drop: float,
    largest: bool = False,
):
    """Algorithm 5."""
    B = as_array_fun(B)
    tau_replace = jnp.asarray(tau_replace, dtype=U.dtype)
    tau_drop = jnp.asarray(tau_drop, dtype=U.dtype)

    i = 0
    v_err = jnp.inf

    def get_u_err(U):
        return frobenius_norm(U.T @ B(U) - jnp.eye(U.shape[1])) / (
            frobenius_norm(B(U)) * frobenius_norm(U)
        )

    while i < 3 and v_err > tau_ortho:
        i += 1
        U = U - V @ (V.T @ B(U))
        U = svqb(B, U, tau_replace, largest=largest)
        j = 1
        u_err = get_u_err(U)

        while j < 3 and u_err > tau_ortho:
            j += 1
            U = svqb_drop(B, U, tau_drop, largest=largest)
            u_err = get_u_err(U)

        v_err = frobenius_norm(V.T @ B(U)) / (frobenius_norm(B(V)) * frobenius_norm(U))
    return U


def lq(X):
    """Get the LQ factorization of matrix X."""
    assert X.ndim == 2
    q, r = jnp.linalg.qr(X.T)
    return r.T, q.T


def _rayleigh_ritz_modified(theta, z_full, nx, nc):
    cx, zo = jnp.split(z_full, (nx,), axis=1)
    z1o = zo[:nx]
    _, q1o = lq(z1o)
    # Different to pseudo-code, but required for shapes described in interface
    q1o = q1o[: nx - nc]
    cp = zo @ q1o.T
    theta_x, theta_o = jnp.split(theta, (nx,))
    Qp = q1o.T
    theta_p = (Qp.T * theta_o) @ Qp
    return theta_x, theta_p, cx, cp


def rayleigh_ritz_modified_ortho(
    S: jnp.ndarray, A: ArrayOrFun, nx: int, nc: int, largest: bool = False,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    SAS = S.T @ as_array_fun(A)(S)
    theta, Z = eigh(SAS, largest=largest)
    return _rayleigh_ritz_modified(theta, Z, nx, nc)


def rayleigh_ritz_modified_non_ortho(
    S: jnp.ndarray,
    A: ArrayOrFun,
    B: ArrayOrFun,
    nx: int,
    nc: int,
    tau_skip: float,
    largest: bool = False,
):
    ns = S.shape[1]
    theta, Z, R, d = _rayleigh_ritz_factorize(S, A, B, largest=largest)
    # pseudo-code has cond(R), but text has cond(R)**-3
    if jnp.linalg.cond(R) ** -3 > tau_skip:
        theta_x = jnp.full((nx,), jnp.nan, S.dtype)
        theta_p = jnp.full((nx - nc, nx - nc), jnp.nan, S.dtype)
        cx = jnp.full((ns, nx), jnp.nan, S.dtype)
        cp = jnp.full((ns, nx - nc), jnp.nan, S.dtype)
        return theta_x, theta_p, cx, cp
    theta_x, theta_p, cx, cp = _rayleigh_ritz_modified(theta, Z, nx, nc)

    def finalize(c):
        return d[:, jnp.array] * (
            jax.scipy.linalg.solve_triangular(R.T, c, lower=False)
        )

    return theta_x, theta_p, finalize(cx), finalize(cp)


def rayleigh_ritz_modified(
    S: jnp.ndarray,
    A: ArrayOrFun,
    B: ArrayOrFun,
    nx: int,
    nc: int,
    use_ortho: bool,
    tau_skip: float,
    largest: bool = False,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray, bool]:
    """
    Algorithm 7.

    Args:
        A:
        B:
        S: [m, ns]
        nx: number of extreme eigenpair approximations to return
        nc: number of converged eigenpairs
        use_ortho: if True, indices S.T @ B(S) == I

    Returns:
        (theta_x, theta_p, cx, cp)

            theta_x: [nx] approximate eigenvalues?
            theta_p: [nx - nc, nx - nc] partial inner products
            cx: [ns, nx]
            cp: [ns, nx - nc]

        Satisfying the following:
            cx.T @ S.T @ A(S) @ cx = jnp.diag(theta_x)
            cp.T @ S.T @ A(S) @ cp = theta_p
            cx.T @ S.T @ B(S) @ cx = eye(nx)
            cp.T @ S.T @ B(S) @ cp = eye(nx - nc)

        If instability is detected, all `nan`s are returned, in which case `use_ortho`
        should be changed.
    """
    if use_ortho:
        return rayleigh_ritz_modified_ortho(S, A, nx, nc, largest=largest)
    return rayleigh_ritz_modified_non_ortho(S, A, B, nx, nc, tau_skip, largest=largest)
