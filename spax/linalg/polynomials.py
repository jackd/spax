import jax
import jax.numpy as jnp

from spax.linalg.utils import as_array_fun
from spax.types import ArrayOrFun


def iterate_chebyshev1(a: ArrayOrFun, p0: jnp.ndarray, p1: jnp.ndarray):
    return p1, 2 * as_array_fun(a)(p1) - p0


@jax.jit
def chebyshev1(a: ArrayOrFun, v: jnp.ndarray, m: int):
    """Chebyshev polynomial T_m(a) @ v."""
    a = as_array_fun(a)

    def cond(state):
        return state[2] < m

    def body(state):
        p0, p1, i = state
        return (*iterate_chebyshev1(a, p0, p1), i + 1)

    def if_0(_):
        return v

    def if_1(_):
        return a(v)

    def otherwise(_):
        state = jax.lax.while_loop(cond, body, (v, a(v), 1))
        return state[1]

    return jax.lax.switch(jnp.minimum(m, 2), (if_0, if_1, otherwise), operand=None)


@jax.jit
def matrix_poly_vector_prod_from_roots(
    roots: jnp.ndarray, A: ArrayOrFun, x: jnp.ndarray
):
    """
    Evaluate matrix-polynomial-vector product with the given roots.

    i.e. `(A - r_0) @ (A - r_1)... @ x`

    The full matrix `A` is never computed.

    Args:
        roots: 1D array of polynomial roots.
        A: `ArrayOrFun` for square matrix at which the polynomial is evaluated.
        x: rhs vector.

    Returns:
        output of the same size as `x`.
    """
    A = as_array_fun(A)
    assert len(roots.shape) == 1

    def body_fun(carry: jnp.ndarray, ri: float):
        el = A(carry) - ri * carry
        return el, el

    res, _ = jax.lax.scan(body_fun, x, roots)
    return res


@jax.jit
def poly_from_roots(roots: jnp.ndarray, x: jnp.ndarray):
    """Evaluate polynomial with given `roots` elementwise on `x`."""
    assert len(roots.shape) == 1

    roots = roots.reshape((-1,) + tuple(1 for _ in x.shape))
    return jnp.prod(x - roots, axis=0)


@jax.jit
def roots_to_coeffs(roots: jnp.ndarray):
    """
    Find the polynomial's coefficients given roots.

    If roots = (r_1, ..., r_n), then the method returns
    coefficients (a_0, a_1, ..., a_n (== 1)) so that
    p(x) = (x - r_1) * ... * (x - r_n)
         = x^n + a_{n-1} * x^{n-1} + ... a_1 * x_1 + a_0
    """
    dtype = roots.dtype
    zero = jnp.zeros((1,), dtype=dtype)
    acc = jnp.ones((1,), dtype=dtype)
    for ri in roots:
        shifted = jnp.concatenate((acc, zero))  # x * acc
        scaled = jnp.concatenate((zero, ri * acc))  # ri * acc
        acc = shifted - scaled
    return acc[-1::-1]


@jax.jit
def matrix_poly_vector_prod_from_coeffs(
    coeffs: jnp.ndarray, A: ArrayOrFun, x: jnp.ndarray
):
    """
    Evaluate matrix-polynomial-vector product with the given coefficients.

    i.e. (coeffs[0] * I + coeffs[1] * A + ... + coeffs[n] A ** n) @ x

    The full matrix `A` is never computed.

    Args:
        roots: 1D array of polynomial roots.
        A: `ArrayOrFun` for square matrix at which the polynomial is evaluated.
        x: rhs vector.

    Returns:
        output of the same size as `x`.
    """
    A = as_array_fun(A)
    assert len(coeffs.shape) == 1

    def body_fun(carry: jnp.ndarray, coeff: float):
        el = A(carry) + coeff * x
        return el, el

    res, _ = jax.lax.scan(body_fun, coeffs[-1] * x, coeffs[-2::-1])
    return res


@jax.jit
def poly_from_coeffs(coeffs: jnp.ndarray, x: jnp.ndarray):
    """Evaluate the polynomial with given `coeffs` element-wise over `x`."""

    def body_fun(carry: jnp.ndarray, coeff: float):
        el = x * carry + coeff
        return el, el

    res, _ = jax.lax.scan(body_fun, jnp.zeros_like(x), coeffs[-1::-1])
    return res
