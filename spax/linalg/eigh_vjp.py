import jax
import jax.numpy as jnp

from spax import ops
from spax.linalg.utils import as_array_fun
from spax.sparse import SparseArray
from spax.types import ArrayOrFun


def project(x, v):
    assert len(x.shape) == 1
    assert len(v.shape) == 1
    return x - v * jnp.dot(v.conj(), x)


def projector(v):
    return jax.tree_util.Partial(project, v=v)


def eigh_rev(grad_w, grad_v, w, v, matmul_fun=jnp.matmul):
    n = w.size
    E = w[jnp.newaxis, :] - w[:, jnp.newaxis]
    vt = v.T

    # grad_a = v (diag(grad_w) + (v^T v.grad / E)) v^T
    #        = v @ inner @ v.T
    inner = jnp.where(jnp.eye(n, dtype=bool), jnp.diag(grad_w), (vt @ grad_v) / E)
    return matmul_fun(v, inner @ vt)


def least_squares_cg(a: ArrayOrFun, b: jnp.ndarray, x0: jnp.ndarray, **kwargs):
    a = as_array_fun(a)
    ata = lambda x: a(a(x).conj()).conj()
    l0, info = jax.scipy.sparse.linalg.cg(ata, a(b.conj()).conj(), x0, **kwargs)
    return l0, info


def eigh_single_rev(
    grad_w, grad_v, w, v, x0, a, outer_impl: jnp.outer, tol: float = 1e-5
):
    """
    Args:
        grad_w: [] gradient w.r.t single eigenvalue.
        grad_v: [m] gradient w.r.t single eigenvector.
        w: single eigenvalue.
        v: single eigenvector.
        a: matmul function.
        x0: initial estimate of least squares solution.
        outer_impl: e.g. `coo.masked_outer_fun`.
        tol: tolerance used in least squares.

    Returns:
        grad_data: shape/dtype of `outer_impl` return value ([m, m] array default).
        x0_: solution to least squares problem solved.
    """
    a = as_array_fun(a)
    g0 = grad_w * outer_impl(v.conj(), v)

    def if_zeros(operand):
        grad_v, w, v, x0 = operand
        del grad_v, v, w
        return x0, jnp.zeros_like(g0)

    def otherwise(operand):
        grad_v, w, v, x0 = operand
        # Amat = (a - wi * jnp.eye(m, dtype=a.dtype)).T
        # Projection operator on space orthogonal to v
        proj = projector(v)

        # Find a solution lambda_0 using least-squares conjugate gradient
        # (A - w*I) @ x = proj(grad_v)
        (l0, _) = least_squares_cg(
            lambda x: a(x.conj()).conj() - x * w, proj(grad_v), x0=x0, atol=0, tol=tol
        )
        # Project to correct for round-off errors
        l0 = proj(l0)
        return l0, -outer_impl(l0, v)

    operand = grad_v, w, v, x0
    x0_, g1 = jax.lax.cond(jnp.all(grad_v == 0), if_zeros, otherwise, operand)
    return g0 + g1, x0_


def eigh_partial_rev(
    grad_w, grad_v, w, v, x0, a, outer_impl=jnp.outer, tol: float = 1e-5
):
    """
    Args:
        grad_w: [k] gradient w.r.t eigenvalues
        grad_v: [m, k] gradient w.r.t eigenvectors
        w: [k] eigenvalues
        v: [m, k] eigenvectors
        a: matmul function
        x0: [m, k] initial solution to (A - w[i]I)x[i] = Proj(grad_v[:, i])
        tol: tolerance used in `jnp.linalg.cg`

    Returns:
        grad_a: [m, m] (or output shape/dtyep of `outer_impl`)
        x0: [m, k]
    """
    a = as_array_fun(a)
    grad_a, x0 = jax.vmap(
        jax.tree_util.Partial(eigh_single_rev, a=a, outer_impl=outer_impl, tol=tol),
        in_axes=(0, 1, 0, 1, 1),
        out_axes=(0, 1),
    )(grad_w, grad_v, w, v, x0)
    grad_a = jnp.sum(grad_a, axis=0)
    return grad_a, x0
    # a = as_array_fun(a)
    # grad_As = []

    # grad_As.append(
    #     jax.vmap(lambda grad_wi, vi: grad_wi * outer_impl(vi.conj(), vi), (0, 1))(
    #         grad_w, v
    #     ).sum(0)
    # )
    # if grad_v is not None:
    #     # Add eigenvector part only if non-zero backward signal is present.
    #     # This can avoid NaN results for degenerate cases if the function
    #     # depends on the eigenvalues only.

    #     def f_inner(grad_vi, wi, vi, x0i):
    #         def if_any(operand):
    #             grad_vi, wi, vi, x0i = operand

    #             # Amat = (a - wi * jnp.eye(m, dtype=a.dtype)).T
    #             adjoint_fun = lambda x: (a(x.conj())).conj() - wi * x

    #             # Projection operator on space orthogonal to v
    #             proj = projector(vi)

    #             # Find a solution lambda_0 using conjugate gradient
    #             (l0, _) = jax.scipy.sparse.linalg.cg(
    #                 adjoint_fun, proj(grad_vi), x0=proj(x0i), atol=0, tol=tol
    #             )
    #             # (l0, _) = jax.scipy.sparse.linalg.gmres(
    #             #     adjoint_fun, proj(grad_vi), x0=proj(x0i)
    #             # )
    #             # l0 = jax.numpy.linalg.lstsq(Amat, P(grad_vi))[0]
    #             # Project to correct for round-off errors
    #             l0 = proj(l0)
    #             return -outer_impl(l0, vi), l0

    #         def if_none(operand):
    #             x0i = operand[-1]
    #             return jnp.zeros_like(grad_As[0]), x0i

    #         operand = (grad_vi, wi, vi, x0i)
    #         # return if_any(operand) if jnp.any(grad_vi) else if_none(operand)
    #         return jax.lax.cond(jnp.any(grad_vi), if_any, if_none, operand)

    #     # x0s = []
    #     # for k in range(grad_v.shape[1]):
    #     #     out = f_inner(grad_v[:, k], w[k], v[:, k], x0[:, k])
    #     #     grad_As.append(out[0])
    #     #     x0s.append(out[1])
    #     # x0 = jnp.stack(x0s, axis=0)
    #     # TODO: revert the above back to using vmap
    #     # it seems to cause issues with jax2tf.convert
    #     grad_a, x0 = jax.vmap(f_inner, in_axes=(1, 0, 1, 1), out_axes=(0, 1))(
    #         grad_v, w, v, x0
    #     )
    #     grad_As.append(grad_a.sum(0))
    # return sum(grad_As), x0


def eigh_partial_rev_sparse(
    grad_w, grad_v, w, v, x0, a: SparseArray, tol: float = 1e-5
):
    """
    Args:
        grad_w: [k] gradient w.r.t eigenvalues
        grad_v: [m, k] gradient w.r.t eigenvectors
        w: [k] eigenvalues
        v: [m, k] eigenvectors
        x0: initial solution to least squares problem.
        data, indices, indptr: csr formatted [m, m] matrix.

    Returns:
        grad_data: gradient of `data` input, same `shape` and `dtype`
        x0_: solution to least squares problem.
    """
    outer_impl = jax.tree_util.Partial(ops.masked_outer, a)
    a = as_array_fun(a)
    grad_data, x0 = eigh_partial_rev(grad_w, grad_v, w, v, x0, a, outer_impl, tol=tol)
    return grad_data, x0
